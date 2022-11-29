# Running the Notebook
Our results can be replicated by running the notebook 'HousePricesFinal.ipynb'.

### Change Encoding Method
By default, the data is encoded with a combination of one-hot and label. 

To change this, go to cell 6 in the notebook:
```python
train_df = InitData("both")
```
The function `InitData` takes an encoding as an argument. The values `"label"`, `"onehot"`, or `"both"` will set the encoding method.
