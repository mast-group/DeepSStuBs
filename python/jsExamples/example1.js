function add() {
	used++;
	if( used > colors.length ) {
		alert( "Only " + colors.length + " percentages are suported." );
		return;
	}
	
	var colorDiv = document.createElement( "div" );
	colorDiv.setAttribute( "class", "color" );
	colorDiv.setAttribute( "id", "div" + used );
	
	var input = document.createElement( "input" );
	input.setAttribute( "class", "percentage" );
	input.setAttribute( "type", "number" );
	input.setAttribute( "id", "color" + used );
	input.setAttribute( "min", "0" );
	input.setAttribute( "max", "100" );
	input.setAttribute( "tabindex", "" + ( 2*(used) + 1 ) );
	input.value = "0";
	
	
	var select = document.createElement( "select" );
	select.setAttribute( "class", "colorSelector" );
	select.setAttribute( "id", "selColor" + used );
	select.setAttribute( "tabindex", "" + ( 2*(used) + 2 ) );
	addColors( select, -1 );
	
	colorDiv.appendChild( input );
	colorDiv.appendChild( select );
	document.getElementById( "form" ).insertBefore( colorDiv, document.getElementById( "buttons" ) );
	
	addHandler();
}
