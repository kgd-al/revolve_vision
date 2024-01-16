#!/bin/bash

out=tmp/memory_debug/
mkdir -p $out

n1=2
n2=6
n3=12

rsync -avzh --progress kevingd@ripper3:data/revolve/identify_debug/ $out

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
     print(f)
     name=f.split('/')[-2]
     print(">", name)
     id, _, end = name.split("-")
     budget, items = end.split("_")
     items = items[:-5]

     id, items, budget = int(id), int(items), int(budget)

     if budget in data:
          data[budget][(id, items)] = f
     else:
          data[budget] = {(id, items): f}

with PdfPages('$out/plots.pdf') as file:
     for budget, series in data.items():
          ids_s = sorted(list(set(id for id, _ in series)))
          items_s = sorted(list(set(items for _, items in series)))
          print(ids_s)
          print(items_s)

          fig, axes = plt.subplots(nrows=len(items_s), ncols=len(ids_s),
                                   sharex=True, sharey=True, layout='constrained')
          for i, id in enumerate(ids_s):
               for j, items in enumerate(items_s):
                    if (folder := series.get((id, items), None)) is not None:
                         df = pd.read_csv(folder + "/memory.log", sep=' ')
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

# outfile=$out/${n1}_${n2}_${n3}.png
# gnuplot <<- EOF -p -
#      set term pngcairo size 1680, 1050;
#      set output '$outfile';
#
#      set multiplot layout 3, 1;
#      set yrange [0:100];
#      set grid;
#
#      plot '$out/memory_$n1.log' u 1 w l title 'python ($n1 items)', \
#           '' u 2 w l title 'system';
#
#      plot '$out/memory_$n2.log' u 1 w l title 'python ($n2 items)', \
#           '' u 2 w l title 'system';
#
#      plot '$out/memory_$n3.log' u 1 w l title 'python ($n3 items)', \
#           '' u 2 w l title 'system';
# EOF
#
# pgrep -a feh | grep "$outfile" > dev.null || feh -.Z --reload 10 $outfile 2>/dev/null &
