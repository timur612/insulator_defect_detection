import os
import pandas as pd

def save_to_csv(res: list, save_path: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, '../..', save_path)
    d = {}
    for i in res:
        x, y, w, h, prob, name = i
        name = name[:-2]
        if name not in d:
            d[name] = [[x, y, w, h, prob]]
        else:
            d[name].append([x, y, w, h, prob])

    bit = []
    for k, v in d.items():
        sub = [k]
        nex = []
        prob = []
        for i in v:
            nex.append([i[0], i[1], i[2], i[3]])
            prob.append(1)  # i[4]
        sub.append(nex)
        sub.append(prob)
        bit.append(sub)

    rs = list(zip(*bit))
    percentile_list = pd.DataFrame(
        {'file_name': rs[0],
         'rbbox': rs[1],
         'probability': rs[2],
         })

    percentile_list.to_csv(save_path+"/result.csv", index=False)
