# Answers

Question 1 : If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

**Approach:**  
I would use techniques like data augmentation (e.g., paraphrasing, back translation) to expand the dataset synthetically. Semi-supervised learning or transfer learning with pre-trained language models can also help leverage the small labeled set effectively. Finally, active learning can be applied to label only the most informative new examples instead of thousands more.

**Mathematical Perspective:**  
Let total required data = `n`. If we have only 200 labeled samples, then missing data = `x = n - 200`. For example, if `n = 1000`, then `x = 800`. Instead of collecting all `x`, we can artificially expand 200 into ~1000 using augmentation and leverage pre-trained embeddings to make the smaller labeled set behave like a larger effective dataset and then can perform imputation.


Question 2 : How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?

Answer : Use balanced, deduplicated data and test fairness with slice metrics and counterfactual checks. Add guardrails in production: safety filtering (toxicity/PII), confidence-based abstain to human review, and continuous monitoring with alerts and fast rollback.



Question 3 : Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

Answers

To generate personalized cold email openers, I would design prompts that include structured context such as the recipient’s name, role, company, and recent achievements. I’d use few-shot examples that demonstrate specific, tailored openers rather than generic ones. Additionally, I’d constrain the model with instructions like “reference only the given context” to keep outputs relevant and avoid filler phrases.


## Comparison and production choice

Summary of results (5-fold Grouped-CV on clean_text):
- SVM (TF-IDF + LinearSVC): Accuracy ≈ 0.965 ± 0.027; Weighted F1 ≈ 0.965 ± 0.028
- Random Forest (TF-IDF + RF): Accuracy ≈ 0.951 ± 0.025; Weighted F1 ≈ 0.950 ± 0.026

Bias–variance (from learning curves):
- Both show variance (Train > CV). Regularization and TF-IDF pruning help.
- SVM tends to exhibit slightly lower variance and higher validation scores.
- RF typically shows a larger train–CV gap unless depth and leaf sizes are constrained.

Confusion (qualitative):
- Both mainly confuse near classes; SVM produces fewer cross-class errors at the same threshold due to a crisper linear boundary.
- RF can be more sensitive to sparse high-dimensional noise features without strong regularization.

Efficiency and ops:
- SVM pipeline is smaller, faster on CPU, and deterministic; easier to serve.
- RF is heavier and slower to train; good with interactions but less efficient here.

Production choice:
- Use TF-IDF + Linear SVM (svm_best_pipeline.pkl).
  - It’s more accurate (higher CV accuracy/F1), simpler, and lower-latency.
  - If calibrated probabilities are needed, wrap with CalibratedClassifierCV; otherwise use decision margins as confidence.
- Keep the tuned RF pipeline (rf_best_pipeline.pkl) as a fallback or for ensemble checks.