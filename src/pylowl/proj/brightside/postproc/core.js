node_size = function(d) {
    return d.active ? (Math.max(Math.min(Math.log(d.lambda_ss),10),0)+2) : 2;
}

set_node_text = function(node, text) {
    var new_child = document.createTextNode(text);
    while (node.hasChildNodes()) {
        node.removeChild(node.firstChild);
    }
    node.appendChild(new_child);
}

add_lambda_ss_text = function(font_size){
    return function(d){ return d.active ?
        "<div id=\"lambda_text\" style=\"font-size: " + font_size +"px\">"
            + Math.round(1000*d.lambda_ss)/1000 + "</div>"
        : "" ;
    }
}

format_words = function(d) {
    var graph_node = lookup_node($graph_root, d.node_placement);
    var s = "";
    for (var w in graph_node.words) {
        if (w > 0) {
            s += " ";
        }
        s += graph_node.words[w]['word'];
    }
    return s;
}

$update_nav_text = function() {
    set_node_text(document.getElementById("window_start_text"), $window_start);
    set_node_text(document.getElementById("window_end_text"), $window_end);
    set_node_text(document.getElementById("num_graphs_text"), $num_graphs);
}

compare_int_arrays = function(array1, array2){
    var equal = true;
    if(array1 != null && array2 != null){
        if(array1.length == array2.length){
            for(var index = 0; index < array1.length; index++){
                if(array1[index] != array2[index]){
                    equal = false;
                }
            }
        }else{
            equal = false;
        }
    }else{
        equal = false;
    }
    return equal;
}

index_in_array = function(innerArray, outerArray){
    var index_in_array = -1;
    for(var arrayx = 0; arrayx < outerArray.length; arrayx++){
        if(compare_int_arrays(innerArray, outerArray[arrayx]) == true){
            index_in_array = arrayx;
            break;
        }
    }
    return index_in_array;
}


qsort = function(a) {
    if (a.length == 0) return [];
 
    var left = [], right = [], pivot = a[0].total_lambda_ss;
 
    for (var i = 1; i < a.length; i++) {
        a[i].total_lambda_ss > pivot ? left.push(a[i]) : right.push(a[i]);
    }
 
    return qsort(left).concat(a[0], qsort(right));
}

$filter_node_first = function(node){
    var end_sort = $root.length;
    var root_index = 0;
    if(node != null){
        if(index_in_array(node.node_placement, $node_sort_array) == -1){
            $node_sort_array[$node_sort_array.length] = node.node_placement;
            for(var graphNum = 0; graphNum < $root.length; graphNum++){
                lookup_node($root[graphNum]["subtree"], node.node_placement).circled = true;
            }
        }else{
            var indexOfNode = index_in_array(node.node_placement, $node_sort_array);
            $node_sort_array.splice(indexOfNode,1);
            for(var graphNum = 0; graphNum < $root.length; graphNum++){
                lookup_node($root[graphNum]["subtree"], node.node_placement).circled = false;
            }
            $filter_node_first(null);
        }
    }
    for(var CirNode = 0; CirNode < $node_sort_array.length; CirNode++){
        for(var graphNum = 0; graphNum < end_sort; graphNum++){
            if(lookup_node($root[graphNum]["subtree"], $node_sort_array[CirNode]).active == true){
                var temp = $root[graphNum];
                $root[graphNum] = $root[root_index];
                $root[root_index] = temp;
                root_index++;
            }
        }
        end_sort = root_index;
        root_index = 0;
    }

    for(var graphNum = 0; graphNum < $root.length; graphNum++){
        var graphLambda = 0;
        for(var nodeInd = 0; nodeInd < $node_sort_array.length; nodeInd++){
            if(lookup_node($root[graphNum]["subtree"], $node_sort_array[nodeInd]).active == true){
                graphLambda += lookup_node($root[graphNum]["subtree"], $node_sort_array[nodeInd]).lambda_ss;
            }
        }
        $root[graphNum].total_lambda_ss = graphLambda;
    }

    var end_first_part_qsort = 0;
    for(var graphNum = 0; graphNum < $root.length; graphNum++){
        var addOne = 1;
        for(var nodeInd = 0; nodeInd < $node_sort_array.length; nodeInd++){
            if(lookup_node($root[graphNum]["subtree"], $node_sort_array[nodeInd]).active == false){
                addOne = 0;
            }
        }
        end_first_part_qsort += addOne;
    }
    
    //NOTE: BESIDES ADDING ELEMENTS TO THESE ARRAYS AND REORDERING THEM, DO NOTHING ELSE. ALIASING CAN BE MEAN.
    var firstArr = [];
    var secondArr = [];
    
    for(var graphNum = 0; graphNum < end_first_part_qsort; graphNum++){
        firstArr[graphNum] = $root[graphNum];
    }
    
    for(var graphNum = end_first_part_qsort; graphNum < $root.length; graphNum++){
        secondArr[graphNum-firstArr.length] = $root[graphNum];
    }
    
    firstArr = qsort(firstArr);
    secondArr = qsort(secondArr);
    
    for(var index = 0; index < secondArr.length; index++){
        firstArr[firstArr.length] = secondArr[index];
    }
    
    for(var graphNum = 0; graphNum < firstArr.length; graphNum++){
        $root[graphNum] = firstArr[graphNum];
    }
}

maximum_Node_Size_rootParent = function(root){
    var min_var_param = 0;
    for(var indiv = 0; indiv < $num_graphs; indiv++){
        var minSubStat = maximum_Node_Size_subtreeChildren(min_var_param, root[indiv]["subtree"]);
        if(minSubStat>min_var_param){
            min_var_param = minSubStat;
        }
    }
    return min_var_param;
}

maximum_Node_Size_subtreeChildren = function(min_var_param, node){
    if(min_var_param < node_size(node)){
        min_var_param = node_size(node);
    }
    for (var c in node.children) {
        var child = node.children[c];
        min_var_param = maximum_Node_Size_subtreeChildren(min_var_param, child);
    }
    return min_var_param;
}

node_placement_rootParent = function(){
	for(var graphNum = 0; graphNum < $root.length; graphNum++){
		node_placement_subtreeChildren($root[graphNum].subtree, 0, []);
	}
}

node_placement_subtreeChildren = function(node, placementInList, currentArray){
    var myArray = [];
    for(var k = 0; k < currentArray.length; k++){
        myArray[k] = currentArray[k];
    }
    myArray[myArray.length] = placementInList;
    node.node_placement = myArray;

    for(var c in node.children){
        var child = node.children[c];
        node_placement_subtreeChildren(child, c, myArray);
    }
}

$update_window_size = function() {
    $window_size = parseInt(document.getElementById("window_size").value, 10);
    $num_windows = Math.ceil($num_graphs / $window_size);
    $shift_window(0);
}

$shift_window = function(delta) {
    $window_start += delta * $window_size;
    if ($window_start < 0) {
        $window_start = 0;
    } else if ($window_start >= $num_graphs) {
        $window_start = ($num_windows - 1) * $window_size;
    }
    $window_end = Math.min($window_start + $window_size, $num_graphs);
}

tree_depth = function(node) {
    var depth = 0;
    for (var c in node.children) {
        var child = node.children[c];
        depth = Math.max(depth, 1 + tree_depth(child));
    }
    return depth;
}

tree_size = function(node) {
    var size = 1;
    for (var c in node.children) {
        var child = node.children[c];
        size += tree_size(child);
    }
    return size;
}

tree_size_per_level = function(node) {
    var sizes = [1];
    for (var c in node.children) {
        var child = node.children[c];
        var c_sizes = tree_size_per_level(child);
        for (var j = 0; j < sizes.length - 1 && j < c_sizes.length; ++j) {
            sizes[j+1] += c_sizes[j];
        }
        for (var j = sizes.length - 1; j < c_sizes.length; ++j) {
            sizes.push(c_sizes[j]);
        }
    }
    return sizes;
}

lookup_node = function(subtree, placement) {
    return lookup_node_helper(subtree, placement.slice(1));
}

lookup_node_helper = function(node, placement) {
    if (placement.length == 0) {
        return node;
    } else {
        return lookup_node_helper(node.children[placement[0]], placement.slice(1));
    }
}

minimum_param_rootParent = function(root, paramName){
    var min_var_param = null;
    for(var indiv = 0; indiv < $num_graphs; indiv++){
        var minSubStat = minimum_param_subtreeChildren(min_var_param, root[indiv]["subtree"], paramName);
        if(min_var_param === null || minSubStat < min_var_param){
            min_var_param = minSubStat;
        }
    }
    return min_var_param;
}

minimum_param_subtreeChildren = function(min_var_param, node, paramName){
    if((min_var_param === null || min_var_param > node[paramName]) && paramName in node){
        min_var_param = node[paramName];
    }
    for (var c in node.children) {
        var child = node.children[c];
        min_var_param = minimum_param_subtreeChildren(min_var_param, child, paramName);
    }
    return min_var_param;
}

maximum_param_rootParent = function(root, paramName){
    var max_var_param = null;
    for(var indiv = 0; indiv < $num_graphs; indiv++){
        var maxSubStat = maximum_param_subtreeChildren(max_var_param, root[indiv]["subtree"], paramName);
        if(max_var_param === null || maxSubStat > max_var_param){
            max_var_param = maxSubStat;
        }
    }
    return max_var_param;
}

maximum_param_subtreeChildren = function(max_var_param, node, paramName){
    if((max_var_param === null || max_var_param > node[paramName]) && paramName in node){
        max_var_param = node[paramName];
    }
    for (var c in node.children) {
        var child = node.children[c];
        max_var_param = maximum_param_subtreeChildren(max_var_param, child, paramName);
    }
    return max_var_param;
}
