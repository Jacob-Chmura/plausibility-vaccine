# plausibility-vaccine

Injecting LLM Knowledge for Event Plausibility

Dataset:
the collection of the data was done by repreatedly prompting CHATGPT-4o to give 100s of items "in a single column, return 1000 distinct items (iteratively prompted)", until about 1000 distinct nouns were reached. for the classification task, the LLM was prompted with "Add another column that categorises the item compared to what it's closest to in terms of sentience. Do to numerically for the following reference points: rock, tree, ant, cat, chimp, man. return a single column". This is an exable for the sentience prompt. it had to be done in small batches repeatedly as the LLM would not maintain attention for the entire dataset
