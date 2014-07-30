$node_size = function(d) {
    return d.active ? (Math.max(Math.min(Math.log(d.lambda_ss),10),0)+2) : 2;
}

var set_node_text = function(node, text) {
    var new_child = document.createTextNode(text);
    while (node.hasChildNodes()) {
        node.removeChild(node.firstChild);
    }
    node.appendChild(new_child);
}

$add_lambda_ss_text = function(font_size){
    return function(d){
        return d.active ?
            ("<div id=\"lambda_text\" style=\"font-size: " + font_size +"px\">" + Math.round(1000*d.lambda_ss)/1000 + "</div>")
            : "" ;
    }
}

$format_words = function(d) {
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

var compare_int_arrays = function(array1, array2){
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

var index_in_array = function(innerArray, outerArray){
    var idx = -1;
    for(var arrayx = 0; arrayx < outerArray.length; arrayx++){
        if(compare_int_arrays(innerArray, outerArray[arrayx]) == true){
            idx = arrayx;
            break;
        }
    }
    return idx;
}

var qsort = function(a) {
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
            $node_sort_array.push(node.node_placement);
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
        firstArr.push(secondArr[index]);
    }
    
    for(var graphNum = 0; graphNum < firstArr.length; graphNum++){
        $root[graphNum] = firstArr[graphNum];
    }
}

$set_subtrees_node_placement = function(){
	for(var graphNum = 0; graphNum < $root.length; graphNum++){
		set_subtree_node_placement($root[graphNum].subtree, 0, []);
	}
}

var set_subtree_node_placement = function(node, placementInList, currentArray){
    var myArray = [];
    for(var k = 0; k < currentArray.length; k++){
        myArray[k] = currentArray[k];
    }
    myArray.push(placementInList);
    node.node_placement = myArray;

    for(var c in node.children){
        var child = node.children[c];
        set_subtree_node_placement(child, c, myArray);
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

$tree_depth = function(node) {
    var depth = 0;
    for (var c in node.children) {
        var child = node.children[c];
        depth = Math.max(depth, 1 + $tree_depth(child));
    }
    return depth;
}

$tree_size = function(node) {
    var size = 1;
    for (var c in node.children) {
        var child = node.children[c];
        size += $tree_size(child);
    }
    return size;
}

$tree_size_per_level = function(node) {
    var sizes = [1];
    for (var c in node.children) {
        var child = node.children[c];
        var c_sizes = $tree_size_per_level(child);
        for (var j = 0; j < sizes.length - 1 && j < c_sizes.length; ++j) {
            sizes[j+1] += c_sizes[j];
        }
        for (var j = sizes.length - 1; j < c_sizes.length; ++j) {
            sizes.push(c_sizes[j]);
        }
    }
    return sizes;
}

var lookup_node = function(subtree, placement) {
    return lookup_node_helper(subtree, placement.slice(1));
}

var lookup_node_helper = function(node, placement) {
    if (placement.length == 0) {
        return node;
    } else {
        return lookup_node_helper(node.children[placement[0]], placement.slice(1));
    }
}

var agg_subtrees_param = function(root, get_param, agg) {
    var curried = function(subtree_container) {
        return agg_subtree_param(subtree_container["subtree"], get_param, agg);
    }
    return agg.apply(null, root.map(curried));
}

var agg_subtree_param = function(node, get_param, agg) {
    var curried = function(n) {
        return agg_subtree_param(n, get_param, agg);
    }
    return agg(
        get_param(node),
        agg.apply(null, node.children.map(curried))
    );
}

$max_subtrees_param = function(root, get_param) {
    return agg_subtrees_param(root, get_param, Math.max);
}

$max_subtree_param = function(node, get_param) {
    return agg_subtree_param(node, get_param, Math.max);
}

$min_subtrees_param = function(root, get_param) {
    return agg_subtrees_param(root, get_param, Math.min);
}

$min_subtree_param = function(node, get_param) {
    return agg_subtree_param(node, get_param, Math.min);
}
