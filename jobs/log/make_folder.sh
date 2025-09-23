#for f in *.OU; do n=$(echo "$f" | grep -o '^[0-9]\+'); d="20250719_${n}"; echo mv "$f" "$d/"; done
for f in *.OU; do n=$(echo "$f" | grep -o '^[0-9]\+'); d="${n}"; mkdir -p "$d"; mv "$f" "$d/"; done
