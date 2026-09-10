TPU_NAME=tn-v5litepod-16-1
ZONE=europe-west4-b
PROJECT_ID=memory-layers
STEP=180
MODEL_NAME=qwen3_mem_embed
REMOTE_PROJECT_DIR=/home/rohunagrawal/memory-layers
CHECKPOINT_DIR=outputs/2026-02-20/23-51-25
LOCAL_PROJECT_DIR=~/memory-layers

# May need to try different workers
WORKER=1

# Get Hydra File
mkdir -p $(dirname ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR}/)

echo "Taring Hydra config..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=${WORKER} \
    --tunnel-through-iap \
    --command="cd ${REMOTE_PROJECT_DIR} && tar -cvzf hydra_config.tar.gz ${CHECKPOINT_DIR}/.hydra"

echo "Copying hydra config..."
gcloud alpha compute tpus tpu-vm scp \
    ${TPU_NAME}:${REMOTE_PROJECT_DIR}/hydra_config.tar.gz \
    ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR}/hydra_config.tar.gz \
    --tunnel-through-iap \
    --worker=${WORKER} \
    --zone=${ZONE} \
    --project=${PROJECT_ID}

echo "Extracting locally..."
tar -xvzf ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR}/hydra_config.tar.gz -C ${LOCAL_PROJECT_DIR}

# 1. SSH in and tar zip the checkpoint dir
echo "Taring ..."
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=${WORKER} \
    --tunnel-through-iap \
    --command="cd ${REMOTE_PROJECT_DIR} && tar -cvzf remote_checkpoint.tar.gz ${CHECKPOINT_DIR}/${MODEL_NAME}/${STEP}/"

# 2. Copy over the checkpoint tar
echo "Copying tar file..."
gcloud alpha compute tpus tpu-vm scp \
    ${TPU_NAME}:${REMOTE_PROJECT_DIR}/remote_checkpoint.tar.gz \
    ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR}/remote_checkpoint.tar.gz \
    --tunnel-through-iap \
    --worker=${WORKER} \
    --zone=${ZONE} \
    --project=${PROJECT_ID}

# 3. Unzip locally
echo "Extracting locally..."
mkdir -p $(dirname ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR})
tar -xvzf ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR}/remote_checkpoint.tar.gz -C ${LOCAL_PROJECT_DIR}

# 4. Cleanup
echo "Cleaning up..."
rm ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR}/remote_checkpoint.tar.gz
rm ${LOCAL_PROJECT_DIR}/${CHECKPOINT_DIR}/hydra_config.tar.gz

gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --worker=${WORKER} \
    --tunnel-through-iap \
    --command="rm ${REMOTE_PROJECT_DIR}/remote_checkpoint.tar.gz && rm ${REMOTE_PROJECT_DIR}/hydra_config.tar.gz"

echo "Done!"