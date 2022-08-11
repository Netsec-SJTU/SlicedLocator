// @Time: 2022.5.19 17:00
// @Author: Bolun Wu
// @Desc: RUN > joern --script gen_graph.scala --params filepath=test.c,saveprefix=tmp

// generate edge and node from filepath and store to saveprefix as JSON
@main def exec(filepath: String, saveprefix: String) = {
   importCode.c(filepath)
   run.ossdataflow // * analyze data flow
   cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> saveprefix + "_edges.json"
   cpg.graph.V.map(node=>node).toJson |> saveprefix + "_nodes.json"
   delete
}
