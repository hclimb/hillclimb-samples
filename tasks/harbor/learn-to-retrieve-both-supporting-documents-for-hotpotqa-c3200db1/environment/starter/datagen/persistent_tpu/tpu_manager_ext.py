'''
Variant of tpu_manager.py that provisions TPUs with external IPs enabled
and does NOT set up a Cloud NAT. Everything else is identical.
'''
import logging
import os
import subprocess
import time
from typing import List

from google.cloud import tpu_v2
from google.api_core.exceptions import NotFound

from work_queue import WorkQueue


class TPUOrchestrator:
    def __init__(self, tpu_id: str, tpu_type: str, zone: str, project_id: str,
                 env_file_path: str, work_queue: WorkQueue,
                 setup_script_path: str, run_command_template: str, work_dir: str):
        self.tpu_id = tpu_id
        self.tpu_type = tpu_type
        self.zone = zone
        self.project_id = project_id
        self.env_file_path = env_file_path
        self.work_queue = work_queue
        self.setup_script_path = setup_script_path
        self.run_command_template = run_command_template
        self.work_dir = work_dir

        self.client = tpu_v2.TpuClient()
        self.qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'

    def get_runtime(self):
        if 'v6e' in self.tpu_type: return 'v2-alpha-tpuv6e'
        elif 'v5p' in self.tpu_type: return 'v2-alpha-tpuv5'
        elif 'v5lite' in self.tpu_type: return 'v2-alpha-tpuv5-lite'
        return 'tpu-ubuntu2204-base'

    def create_tpu(self):
        logging.info(f"[{self.tpu_id}] Requesting QueuedResource creation (external IP enabled)...")
        node_spec = tpu_v2.QueuedResource.Tpu.NodeSpec(
            parent=f'projects/{self.project_id}/locations/{self.zone}',
            node_id=self.tpu_id,
            node=tpu_v2.Node(
                accelerator_type=self.tpu_type,
                runtime_version=self.get_runtime(),
                network_config=tpu_v2.NetworkConfig(enable_external_ips=True),
            ),
        )
        qr = tpu_v2.QueuedResource(
            tpu=tpu_v2.QueuedResource.Tpu(node_spec=[node_spec]),
        )
        # QueuedResource.Spot() is an empty proto message; proto3 drops empty
        # messages during serialization so the spot field never reaches GCP.
        # SetInParent() forces the oneof field to be explicitly marked as set.
        qr._pb.spot.SetInParent()
        operation = self.client.create_queued_resource(
            parent=f"projects/{self.project_id}/locations/{self.zone}",
            queued_resource_id=self.tpu_id,
            queued_resource=qr,
        )
        operation.result()

    def delete_tpu(self):
        logging.info(f"[{self.tpu_id}] Deleting QueuedResource...")
        request = tpu_v2.DeleteQueuedResourceRequest(name=self.qr_name, force=True)
        try:
            operation = self.client.delete_queued_resource(request=request)
            operation.result()
        except Exception as e:
            logging.warning(f"Error deleting TPU (might already be deleted): {e}")

        while True:
            try:
                self.client.get_queued_resource(name=self.qr_name)
                time.sleep(5)
            except NotFound:
                break

    def wait_for_ssh(self):
        logging.info(f"[{self.tpu_id}] Polling SSH daemon until ready...")
        cmd = [
            "gcloud", "--quiet", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.tpu_id,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--tunnel-through-iap",
            "--worker=all",
            "--ssh-flag=-o ConnectTimeout=10",
            "--ssh-flag=-o BatchMode=yes",
            "--command", "echo SSH_READY"
        ]

        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and "SSH_READY" in result.stdout:
                    logging.info(f"[{self.tpu_id}] SSH daemon is fully responsive!")
                    return True
                else:
                    logging.info(f"[{self.tpu_id}] SSH poll failed with code {result.returncode}, retrying...")
            except subprocess.TimeoutExpired:
                logging.info(f"[{self.tpu_id}] SSH poll timed out, retrying...")
            time.sleep(10)

        logging.warning(f"[{self.tpu_id}] SSH polling timed out after 5 minutes.")
        return False

    def wait_for_active(self):
        while True:
            try:
                qr = self.client.get_queued_resource(name=self.qr_name)
                state = qr.state.state.name

                if state == 'ACTIVE':
                    logging.info(f"[{self.tpu_id}] Google API reports TPU is ACTIVE.")
                    if self.wait_for_ssh():
                        return True
                    else:
                        logging.warning(f"[{self.tpu_id}] SSH failed after ACTIVE. Recreating...")
                        self.delete_tpu()
                        self.create_tpu()
                elif state in ('FAILED', 'SUSPENDED'):
                    logging.info(f"[{self.tpu_id}] TPU is in terminal state '{state}'. Recreating...")
                    self.delete_tpu()
                    self.create_tpu()
                else:
                    logging.debug(f"[{self.tpu_id}] TPU state is {state}. Waiting...")
                    time.sleep(15)

            except NotFound:
                logging.info(f"[{self.tpu_id}] TPU not found. Creating a new QueuedResource...")
                self.create_tpu()
                time.sleep(15)

    def kill_remote_processes(self):
        logging.info(f"[{self.tpu_id}] Killing remote processes on TPU...")
        cmd = [
            "gcloud", "--quiet", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.tpu_id,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--tunnel-through-iap",
            "--worker=all",
            "--ssh-flag=-o ConnectTimeout=10",
            "--command", "pkill -9 -f 'eval_worker|eval\\.py|python' 2>/dev/null || true"
        ]
        try:
            subprocess.run(cmd, timeout=30)
            logging.info(f"[{self.tpu_id}] Remote processes killed.")
        except KeyboardInterrupt:
            logging.warning(f"[{self.tpu_id}] Kill interrupted — remote process may still be running.")
        except Exception as e:
            logging.warning(f"[{self.tpu_id}] Could not kill remote processes: {e}")

    def run_ssh_command(self, script_content: str) -> int:
        cmd = [
            "gcloud", "--quiet", "alpha", "compute", "tpus", "tpu-vm", "ssh", self.tpu_id,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--tunnel-through-iap",
            "--worker=all",
            "--ssh-flag=-o ServerAliveInterval=30",
            "--ssh-flag=-o ServerAliveCountMax=3",
            "--ssh-flag=-o ConnectTimeout=30",
            "--command", script_content
        ]
        logging.info("Executing task via SSH...")
        try:
            result = subprocess.run(cmd)
            logging.info(f"SSH execution finished with exit code {result.returncode}")
            return result.returncode
        except Exception as e:
            logging.error(f"SSH execution failed: {e}")
            return 255

    def process_chunk(self, chunk: List[int]) -> bool:
        env_vars = ""
        if os.path.exists(self.env_file_path):
            with open(self.env_file_path, "r") as f:
                env_vars = f.read()
        if "GIT_BRANCH" in os.environ:
            env_vars += f"\nGIT_BRANCH={os.environ['GIT_BRANCH']}"

        setup_script_content = ""
        if self.setup_script_path and os.path.exists(self.setup_script_path):
            with open(self.setup_script_path, "r") as f:
                setup_script_content = f.read()

        chunk_str = ",".join(map(str, chunk))
        resolved_run_command = self.run_command_template.replace("{CHUNKS}", chunk_str).replace("{CHUNK}", chunk_str)

        import textwrap
        ssh_script_template = textwrap.dedent("""\
            #!/bin/bash
            set -eo pipefail

            cat << 'EOF' > /tmp/.tpu_env
            {env_vars}
            EOF
            set -a
            source /tmp/.tpu_env || true
            set +a

            echo "=== INITIALIZING TPU {tpu_id} ==="

            if [ ! -f "$HOME/.initialized" ]; then
                echo "First time setup: Executing provided setup script..."
            {setup_script_content}

                touch $HOME/.initialized
            else
                echo "Already initialized. Skipping setup script..."
            fi

            source $HOME/.local/bin/env || true

            if [ -d "$HOME/{work_dir}" ]; then
                cd "$HOME/{work_dir}"
            else
                echo "Warning: Work directory $HOME/{work_dir} not found. Staying in $HOME"
            fi

            source .venv/bin/activate || true

            cp /tmp/.tpu_env .env || true

            git pull || true

            echo "=== TPU {tpu_id} RUNNING CHUNK {{ {chunk_str} }} ==="
            {resolved_run_command}
        """)

        ssh_script = ssh_script_template.format(
            tpu_id=self.tpu_id,
            setup_script_content=textwrap.indent(setup_script_content, "    ") if setup_script_content else "",
            work_dir=self.work_dir,
            env_vars=env_vars,
            chunk_str=chunk_str,
            resolved_run_command=resolved_run_command
        )

        exit_code = self.run_ssh_command(ssh_script)

        if exit_code == 0:
            return True
        elif exit_code == 255:
            logging.warning("SSH connection lost (exit code 255). Likely preempted. Marking as retryable.")
            return False
        else:
            logging.warning(f"Script failed with code {exit_code}. Marking as failed, will retry on fresh VM.")
            return False

    def run(self):
        logging.info("Orchestrator loop started.")
        while True:
            if self.work_queue.is_done():
                logging.info("No more work in queue. Shutting down cleanly.")
                break

            chunk = self.work_queue.pop_pending()
            if chunk is None:
                logging.info("Queue is empty but work is still ongoing. Sleeping...")
                time.sleep(30)
                continue

            logging.info(f"Popped chunk {chunk} from queue.")
            self.work_queue.print_status()

            self.wait_for_active()

            success = self.process_chunk(chunk)

            if success:
                logging.info(f"Successfully processed chunk {chunk}.")
                self.work_queue.mark_completed(chunk)
            else:
                logging.warning(f"Failed to process chunk {chunk}. Putting back in pending queue.")
                self.work_queue.mark_failed(chunk)
                time.sleep(15)
