for f in $(awk -F/ '{print $NF}' filelist.txt); do
  unzip "$f"
done

