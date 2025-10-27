from graphviz import Digraph

def visualize(self, output_path):
    r"""A visualization tool for AbstractNetwork."""

    nodes = list(self.nodes())
    # Create a directed graph
    dot = Digraph(format='png', engine='dot')
    # Add nodes with optional attributes
    for node in nodes:
        ori_name = getattr(node, 'ori_name', None)
        ori_name = ori_name.split('/')[-1] if ori_name else None
        ori_name_line = f'<TR><TD><FONT FACE="Arial" COLOR="black">{ori_name}</FONT></TD></TR>' if ori_name else ''
        
        label = f"""<
            <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0">
                <TR><TD><FONT FACE="Courier" COLOR="black" >{node.name}</FONT></TD></TR>
                {ori_name_line}
                <TR><TD><FONT FACE="Courier" COLOR="blue" >{node.__class__.__name__}</FONT></TD></TR>
                <TR><TD><FONT FACE="Courier" COLOR="black" >in: {tuple(node.input_shape) if node.input_shape is not None else None}</FONT></TD></TR>
                <TR><TD><FONT FACE="Courier" COLOR="black" >out: {tuple(node.output_shape) if node.output_shape is not None else None}</FONT></TD></TR>
            </TABLE>
        >"""
        if node.__class__.__name__ in ["AbstractParameter", "AbstractConstant", "AbstractBuffer"]: 
            dot.node(node.name, label=label, shape="rectangle", style='filled', fillcolor=node.color)
        elif node.__class__.__name__ == "AbstractInput":
            dot.node(node.name, label=label, shape="doubleoctagon", fillcolor=node.color, style='filled')
        else:
            dot.node(node.name, label=label, shape="rectangle", fillcolor=node.color, style='filled')
            
        # edges
        for inp in node.inputs:
            dot.edge(inp.name, node.name)
    
    # Render graph
    dot.render(output_path, cleanup=True)
    print(f"Graph saved to {output_path}.png")
        
        