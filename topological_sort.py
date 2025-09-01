
# edge goes from smaller to bigger
def poset_to_edges(V, mycmp):
	for v1 in V:
		for v2 in V:
			if mycmp(v1,v2):
				yield (v2, v1)
				
def iterable_to_edge_dict(list_of_edges):
	ed = {}
	for v1, v2 in list_of_edges:
		ed.setdefault(v1,set()).add(v2)
	return ed

def iter_edges(edge_dict):
	for v1, eout in edge_dict.items():
		for v2 in eout:
			yield (v1,v2)
			
def reverse_edge_dict(ed):
	def reverse_edges(edge_iterable):
		for v1, v2 in edge_iterable:
			yield v2, v1
	return iterable_to_edge_dict(reverse_edges(iter_edges(ed)))

def topological_sort(Graph):
	V,E = Graph[0], Graph[1].copy()

	in_egde_count = {}
	for v1, edges_out in E.items():
		for v2 in edges_out:
			in_egde_count[v2] = in_egde_count.get(v2,0) + 1
	zero_in = [ v for v in V if v not in in_egde_count ]

	if len(zero_in) == 0:
		return None

	topo_order = []

	while len(zero_in):
		vz = zero_in[0]
		del zero_in[0]
		topo_order.append(vz)
		edges_out_vz = E.get(vz, None)
		if edges_out_vz != None:
			for v_out_vz in edges_out_vz:
				in_egde_count[v_out_vz] -= 1
				if in_egde_count[v_out_vz] == 0:
					zero_in.append(v_out_vz)

	return topo_order

if __name__ == "__main__":
	print(topological_sort((range(1,5), dict([(1,set([2,3])),(2,set([4])),(3,set([4]))]))) )
	print(topological_sort((range(1,7), iterable_to_edge_dict([(1,2),(1,3),(2,4),(3,4),(6,7)]))) )
	print(topological_sort((range(3), dict([(1,set([2])),(2,set([0])),(0,set([1]))]))) )
