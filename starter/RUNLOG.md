# RUNLOG.md

I decided not to focus on the tokenization component. Instead, I focused on improving the model architecture, optimization setup, learning rate schedule, and other training details.

All runs were trained under the assignment constraints: CPU only, at most 2,000 optimizer steps, at most 2,000,000 parameters, and using only the provided `train_corpus.txt`.

## Baseline
The baseline run achieved a dev bpb of **2.3718**.

## Run 1
**Hypothesis:** Weight tying should improve parameter efficiency and lower bpb.

**What changed:** I enabled weight tying.

**Dev bpb:** 2.3718 -> **2.4122**

**Conclusion:** This change made performance worse. I expected bpb to decrease, but it increased instead, so weight tying alone was not helpful in this setup.

## Run 2
**Hypothesis:** Increasing model depth might improve representation quality.

**What changed:** I increased the number of layers from 4 to 6 while keeping weight tying enabled.

**Dev bpb:** 2.4122 -> **2.4060**

**Conclusion:** This was a very small improvement over Run 1, but it was still worse than the baseline. Increasing depth alone did not solve the main problem.

## Run 3
**Hypothesis:** Better parameter initialization might stabilize training and improve convergence.

**What changed:** I used standard Gaussian initialization for linear and embedding layers. Linear weights were initialized from a normal distribution with mean 0 and standard deviation 0.02, biases were set to zero, and embedding weights were initialized the same way. I also removed weight tying and restored the layer count to the default value of 4.

**Dev bpb:** 2.4060 -> **2.2280**

**Conclusion:** This was a clear improvement. Removing weight tying and using cleaner initialization helped training substantially.

## Run 4
**Hypothesis:** AdamW might improve optimization through better handling of weight decay.

**What changed:** I replaced Adam with AdamW.

**Dev bpb:** 2.2280 -> **2.3368**

**Conclusion:** This change hurt performance, so I removed AdamW for the next experiment. At this point, Run 3 remained the best run.

## Run 5
**Hypothesis:** A deeper model might work better once AdamW is removed.

**What changed:** I removed AdamW and increased the number of layers to 6.

**Dev bpb:** 2.2280 -> **2.2986**

**Conclusion:** This was worse than Run 3. Increasing the number of layers was not the main source of improvement, so the bottleneck likely lay elsewhere.

## Run 6
**Hypothesis:** RoPE positional encoding might improve sequence modeling.

**What changed:** I replaced the earlier positional handling with RoPE positional encoding.

**Dev bpb:** 2.2986 -> **2.3338**

**Conclusion:** RoPE did not help in this setting and made the score worse.

## Run 7
**Hypothesis:** A larger embedding dimension might improve model capacity more effectively than adding layers.

**What changed:** I increased the embedding dimension to 192.

**Dev bpb:** 2.3338 -> **2.1916**

**Conclusion:** This was a strong improvement. It suggested that embedding dimension was a more important bottleneck than model depth.

## Run 8
**Hypothesis:** A better learning rate schedule could improve optimization within the fixed 2,000-step budget.

**What changed:** I introduced a learning rate schedule with a short warmup phase followed by cosine decay. The learning rate gradually increased at the beginning to stabilize early training, and then decreased smoothly over the rest of training to allow better convergence.

**Dev bpb:** 2.1916 -> **2.2160**

**Conclusion:** This schedule was slightly worse than Run 7, so it did not improve on the current best configuration.

## Run 9
**Hypothesis:** Gradient clipping and a larger batch size might stabilize updates and improve training.

**What changed:** I removed the previous learning rate schedule, added gradient clipping, and increased the batch size to 12.

**Dev bpb:** 2.2160 -> **2.1174**

**Conclusion:** This was a clear improvement. Gradient clipping and a slightly larger batch size helped training become more stable and effective.

## Run 10
**Hypothesis:** A different learning rate schedule might work better than the cosine schedule.

**What changed:** I used a schedule with three phases: a short warmup at the beginning, a long constant-learning-rate middle phase, and a final decay near the end of training. This was meant to combine early stability with sustained learning and a gentle finish.

**Dev bpb:** 2.1174 -> [result used in next run]

**Conclusion:** This schedule was promising enough to keep and test further with additional changes.

## Run 11
**Hypothesis:** Increasing batch size further and reintroducing weight tying might improve efficiency.

**What changed:** I increased the batch size to 16 and set weight tying to `True`.

**Dev bpb:** 2.1174 -> **1.9946**

**Conclusion:** This was a major improvement and the first run to go below 2.0 bpb. In this stronger training setup, weight tying became beneficial.

## Run 12
**Hypothesis:** AdamW might work better once the rest of the training setup is stronger.

**What changed:** I reintroduced AdamW and increased the default learning rate from `3e-4` to `6e-4`.

**Dev bpb:** 1.9946 -> **1.9704**

**Conclusion:** This was the best result overall. AdamW did not help earlier, but it became effective once combined with the stronger configuration from the later runs.

## Best Result
My best dev bpb was **1.9704**.

## Final Takeaways
The main improvements came from increasing embedding dimension, adding gradient clipping, increasing batch size, refining the learning rate schedule, and finally using AdamW with a higher learning rate. In contrast, simply increasing depth or switching positional encoding did not help much in this budget-constrained setting. 