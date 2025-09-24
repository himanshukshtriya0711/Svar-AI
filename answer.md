# Answers

Question 1: If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

With only 200 labels, I’d do three things:
- Create more training signals with light data augmentation (paraphrasing, back‑translation).
- Leverage unlabeled data via transfer learning or semi‑supervised learning.
- Use active learning so we label only the most informative new examples instead of thousands at random.

Question 2: How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?

Start with balanced, deduplicated data and check fairness by looking at metrics across slices and with simple counterfactual swaps (e.g., swapping identity terms). In production, add guardrails: safety filters (toxicity/PII), low‑confidence “abstain to human” routing, and continuous monitoring with alerts and fast rollback.

Question 3: Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non‑generic?

Give the model structured context (name, role, company, recent work) and a few clear, specific examples. Add instructions like “use only the provided context,” set length/format constraints, and ask for concrete details to avoid generic fluff.

## Comparison and production choice

Summary of results (5‑fold Grouped‑CV on clean_text):
- SVM (TF‑IDF + LinearSVC): Accuracy ≈ 0.965 ± 0.027; Weighted F1 ≈ 0.965 ± 0.028
- Random Forest (TF‑IDF + RF): Accuracy ≈ 0.951 ± 0.025; Weighted F1 ≈ 0.950 ± 0.026

Bias–variance (from learning curves):
- Both show some overfitting (Train > CV). Regularization and TF‑IDF pruning reduce the gap.
- SVM has a slightly smaller gap and higher validation scores.
- RF typically needs tighter limits (max depth, min samples per leaf) to control variance.

Confusion (qualitative):
- Both mainly confuse neighboring classes, but SVM makes fewer cross‑class mistakes at the same threshold thanks to a crisper linear boundary.
- RF is more sensitive to sparse, high‑dimensional noise unless strongly regularized.

Efficiency and ops:
- SVM pipeline is small, CPU‑friendly, deterministic, and easy to serve.
- RF is heavier and slower to train/serve for this sparse TF‑IDF setting.

Production choice:
- Ship TF‑IDF + Linear SVM (`svm_best_pipeline.pkl`) for best accuracy/F1, lower latency, and simpler operations.
  - If you need calibrated probabilities, wrap it with `CalibratedClassifierCV`; otherwise, use decision margins as confidence.
- Keep the tuned RF pipeline (`rf_best_pipeline.pkl`) as a fallback or for occasional ensemble checks.