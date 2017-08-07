# coding: utf-8
for c in data.columns:
    try:
        data[c] = data[c].apply(lambda x: x.replace('/notebooks/USDCN-project3_Behavioural_Cloning/data/', ''))
    except:
        pass
