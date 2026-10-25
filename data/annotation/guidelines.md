# Annotation guidelines

Label each review with exactly ONE of: `bug report`, `feature request`, `other`.
Decide from the review text alone. Do not consult the original dataset, model outputs, or the other annotator.

Class definitions (verbatim from the evaluation prompt):

- **bug report**: report a problem, such as faulty behavior of the application. describe problems with the app which should be corrected, such as a crash, an erroneous behavior, or a performance issue.
- **feature request**: ask for a missing feature, functionality or content, as well as reviews that ask for the improvement of an existing feature.
- **other**: users express general appreciation with the application. It focuses on general judgment. They are simple text reflections of the numeric star rating. Ratings are less informative as they only include praise, dispraise, a distractive critique, or a dissuasion. Reviews where users describe workarounds, use cases and scenarios involving the app

Set `multi_intent` = 1 only when the reviewer expresses two intents in separate clauses (e.g. a crash AND a request).
Put the dominant intent in `label` anyway. Use `note` for anything a third reader should know.
Work in queue order; do not re-sort. Take breaks every 100 items. Do not discuss items until both queues are complete.
