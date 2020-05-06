for f in *.txt; do echo -n $f;tail -1 $f;done | sed -e 's/.pth.txt\[/,/g' -e 's/ /,/g' -e 's/\]/,/g' -e 's/,,/,/g' -e 's/,,/,/g' > exp_res.csv
