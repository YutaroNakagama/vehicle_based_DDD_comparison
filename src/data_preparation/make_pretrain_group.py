# make_general_subjects.py
with open('config/subjects/subject_list.txt') as fin, open('config/subjects/target_groups.txt') as fin2, open('config/subjects/general_subjects.txt', 'w') as fout:
    # Extract only the first 10 subjects from the first group
    finetune_targets = set(fin2.readline().split())
    for subj in (line.strip() for line in fin if line.strip()):
        if subj not in finetune_targets:
            print(subj, file=fout)

