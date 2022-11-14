from log_handling import LogLoader
import datasets



loader = LogLoader(
    logdir='logs',
    tasks='paper-full',
    model_families=['BIG-G T=0','BIG-G-sparse'],
    # model_sizes=['128b'],
    query_types=['multiple_choice'],
    shots=[3],
    include_unknown_shots=True,
    exclude_faulty_tasks=True,
)
bigdf = datasets.to_dataframe(loader)

bigdf.to_csv('3shot.csv')


