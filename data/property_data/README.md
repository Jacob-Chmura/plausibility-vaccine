### Property Data

______________________________________________________________________

This data was generated as follows.

**Model**: [gpt-4o](https://platform.openai.com/docs/models#gpt-4o)

**Prompt**:

<blockquote>

1. In a single column, return 1000 distinct items
2. Add another column that numerically categorizes each item according to it's size with respect to the reference points: ant, watch, book, cat, person, jeep, house, stadium.
3. Add another column that numerically categorizes each item according to it's physical phase with respect to the reference points: smoke, milk, wood, diamond.
4. Add another column that numerically categorizes each item according to it's sentience with respect to the reference points: rock, tree, ant, cat, chimp, man.
5. Add another column that numerically categorizes each item according to it's weight with respect to the reference points: watch, book, dumbbell, man, jeep, house, stadium.
6. Add another column that numerically categorizes each item according to it's rigidity with respect to the reference points: water, skin, leather, wood, metal.

Generate the data in CSV format. Write one item per line with the item and it's appropriate categories seperated by a commas.

</blockquote>

_Note_: the LLM had to be repeatedly prompted until about 1000 distinct items were generated. To maintain attention, the property attributes were generated in baches of 100 items.
_Note_: some indices in the size category were registered as "out of distribution", as there were points 1-9. That is due to the fact that size was too challenging for the LLM to categorise, hence the categories of "smaller than ant" and "larger than stadium" were added, and the other categories were made "in between" reference points. THis results in 9 instead of 8 categories.
