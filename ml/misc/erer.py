normal_count   = train_df[train_df['label']==0]['corp_code'].nunique()
target_count   = normal_count // 3
generate_count = max(target_count - bankrupt_count, 0)