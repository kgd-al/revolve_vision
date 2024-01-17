#!/bin/bash

if [ -d "$1" ]
then
     out=$1
else
     out=tmp/memory_debug/
     mkdir -p $out

     rsync -avzh --progress kevingd@ripper3:data/revolve/identify_debug/ $out
fi

outfile=$out/debug_memory_plots.pdf

source ../venv/bin/activate
python3 - <<EOF

import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

folders=sorted(glob.glob('$out/*X*items/'), key=len)
print(folders)

data = {}
for f in folders:
     name=f.split('/')[-2]
     id, _, end = name.split("-")
     budget, items = end.split("_")
     items = items[:-5]

     id, items, budget = int(id), int(items), int(budget)

     if budget in data:
          data[budget][(id, items)] = f
     else:
          data[budget] = {(id, items): f}

print("Collected")
with PdfPages('$outfile') as file:
     for budget, series in sorted(data.items()):
          ids_s = sorted(list(set(id for id, _ in series)))
          items_s = sorted(list(set(items for _, items in series)))

          fig, axes = plt.subplots(nrows=len(items_s), ncols=len(ids_s),
                                   sharex=True, sharey=True, layout='constrained')
          for i, id in enumerate(ids_s):
               for j, items in enumerate(items_s):
                    if (folder := series.get((id, items), None)) is not None:
                         df = pd.read_csv(folder + "/memory.log", sep=' ')
                         if len(ids_s) == 1 and len(items_s) == 1:
                              ax = axes
                         elif len(ids_s) == 1:
                              ax = axes[j]
                         elif len(items_s) == 1:
                              ax = axes[i]
                         else:
                              ax = axes[j,i]
                         ax.grid()

                         pyt, = ax.plot(df.index, df.iloc[:, 0], label='python')
                         sys, = ax.plot(df.index, df.iloc[:, 1], label='system')
                         ax.set_title(folder.split("/")[-2])

          fig.supxlabel("Evaluation")
          fig.supylabel("Memory (%)")

          fig.legend((pyt, sys), ("Python", "System"),
                     loc='outside upper center', ncols=2)
#           fig.tight_layout()
          file.savefig(fig)

EOF

[ $? -eq 0 ] && okular --unique $outfile &
echo
