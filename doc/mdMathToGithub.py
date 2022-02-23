""" Translate a markdown file with dollar-sign-enclosed Latex expressions to a markdown file suitable for a Github README with those expressions displayed appropriately """ 

with open('README_raw.md', 'r') as f:
    t = f.read()
reassembled_txt = ""
txt_split_by_dollars = t.split('$')
for dollar_idx, txt_part in enumerate(txt_split_by_dollars[:-1]):
    if dollar_idx % 2 == 0:
        reassembled_txt += txt_part + ' <img src="https://render.githubusercontent.com/render/math?math='
    else:
        reassembled_txt += txt_part + '">'
reassembled_txt += txt_split_by_dollars[-1]                                                
with open('../README.md', 'w') as f:
    f.write(reassembled_txt)