
**The problem:** When someone's heart beats abnormally (called an arrhythmia), doctors currently have to look at ECG readings (heart signal graphs) by eye to spot the problem. This is slow, and different doctors might read the same signal differently. It also doesn't work well for things like smartwatches that need to monitor your heart constantly and automatically — no doctor is watching those in real time.

**What you're building:** You're teaching computers to automatically spot these abnormal heartbeats instead of relying on a human to read the graph. To do this properly, you're building **three different "detector" programs** and comparing which one works best:

1. **Model 1 (the "old-school" method):** You manually calculate some useful numbers from the heartbeat signal (like how far apart beats are, or the shape of the wave) and feed those numbers into a classic algorithm (SVM or Random Forest) that learns to say "normal" or "abnormal."
2. **Model 2 (1D-CNN):** Instead of you calculating features by hand, you feed the raw heartbeat signal directly into a neural network, and it learns on its own what patterns matter.
3. **Model 3 (CNN-LSTM):** Same idea as Model 2, but it also pays attention to the *sequence* of beats over time — because some arrhythmias only show up as a pattern across several beats in a row, not just one beat alone.

**The side experiment:** For Model 2 (the CNN), there are "settings" (hyperparameters — like how many filters it uses, how fast it learns) that need tuning to get the best performance. Normally people tune these by hand or with standard tools (Adam/SGD). You're instead trying two nature-inspired search algorithms — Grey Wolf Optimizer and Whale Optimization Algorithm (yes, literally inspired by how wolves and whales hunt) — to see if they can find better settings automatically than the standard approach.

**How you judge everything fairly:** All three models are trained and tested on the exact same data, split the same way, and scored with the same metrics (accuracy, precision, recall, F1-score, confusion matrix) — so it's an apples-to-apples comparison.

**Why it matters:** You end up with a clear answer to two useful questions — (1) is it worth using a complex deep learning model instead of a simpler classical one for this task, and (2) is it worth using a fancy nature-inspired tuning method instead of the standard one? Both are practical, real-world trade-off questions, not just "which model gets the highest number."
