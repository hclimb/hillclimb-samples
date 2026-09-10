Help a small maze solving model learn to solve new 17x17 mazes with the shortest correct paths. Edit `/environment/starter/maze_task/candidate.py` to choose which training examples the model should train on and how much weight the examples should get. Please keep the model architecture, optimizer, and number of training steps the same. Only modify `candidate.py`. 

Read `/environment/starter/maze_task/README.md` for information regarding the submission API, resource budgets, and reward scoring. 

To test your method, `/environment/starter/maze_task/public_test.sh` with one of these options:

- `--mode contract`: Quick CPU-only check using four training mazes that verifies `candidate.py` returns valid training examples.
- `--mode quick`: Trains and scores your method using smaller maze groups and fewer training steps. Use it for fast feedback while developing. Use --mode full to evaluate your final method.
- `--mode full`: Trains and scores your method using three separate public maze groups. Each group has 192 training mazes and 96 separate evaluation mazes.
- `--mode full --validation-fold 1`: Same as full but uses a public test set of three maze groups that are separate from the default public test set.

In full mode, we train the model twice for each maze group using your selected examples. Both runs start from the same saved model, but see the examples in different orders (seeds). Those orders are fixed for repeatability. Your method's score is the average score across all groups and both seeds. We test the baseline the same way and show its score separately.

You have a six-hour working window and must use essentially all of it. Do not stop working while more than 5 minutes remain. Run `date` at the start and periodically thereafter. An improvement or passing result is not completion; keep doing whatever work you believe is most promising.