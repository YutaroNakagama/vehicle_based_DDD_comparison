# make_target_groups.py
with open('subject_list.txt') as fin, open('target_groups.txt', 'w') as fout:
    subjects = [line.strip() for line in fin if line.strip()]
    for i in range(0, len(subjects), 10):
        group = ' '.join(subjects[i:i+10])
        print(group, file=fout)


