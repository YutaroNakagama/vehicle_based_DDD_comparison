# make_general_subjects.py
with open('subject_list.txt') as fin, open('target_groups.txt') as fin2, open('general_subjects.txt', 'w') as fout:
    # 1行目（1グループ目）の10人だけ抜く場合
    finetune_targets = set(fin2.readline().split())
    for subj in (line.strip() for line in fin if line.strip()):
        if subj not in finetune_targets:
            print(subj, file=fout)

