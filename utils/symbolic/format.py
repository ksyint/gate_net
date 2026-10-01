def symbolic_equations(circuit):
    """Name every DAG node once, avoiding exponential expression expansion."""
    lines = ["# Binary inputs x[0], ..., x[input_dim - 1]."]
    previous = [f"x[{j}]" for j in range(circuit["input_dim"])]
    for depth, layer in enumerate(circuit["layers"]):
        current = []
        for j, (left, right, gate) in enumerate(zip(layer["left"], layer["right"], layer["gate"])):
            a, b = previous[left], previous[right]
            expressions = ("False", f"({a} and {b})", f"({a} and not {b})", a,
                           f"(not {a} and {b})", b, f"({a} != {b})", f"({a} or {b})",
                           f"not ({a} or {b})", f"({a} == {b})", f"not {b}",
                           f"({a} or not {b})", f"not {a}", f"(not {a} or {b})",
                           f"not ({a} and {b})", "True")
            name = f"layer{depth}_{j}"
            lines.append(f"{name} = {expressions[gate]}")
            current.append(name)
        previous = current
    lines.append("# Final logic features: " + ", ".join(previous))
    lines.append("# Class logits use the linear head saved in circuit.json.")
    return "\n".join(lines) + "\n"
