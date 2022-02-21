from torch.utils.data.dataloader import default_collate

""" Esto lo saqué del código de JuanPablo
"""
def my_collate(batch):
	batch = filter(lambda img: img is not None, batch)
	return default_collate(list(batch))