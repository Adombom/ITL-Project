// ******************************************************************************************
// * This project is licensed under the GNU Affero GPL v3. Copyright © 2019                 *
// ******************************************************************************************

private ["_distFromBuildings", "_generateRandCoords", "_awayFromBuilding", "_generatedCoords", "_awayFrmBldgBool"];

//Distance camera must be away from buildings
_distFromBuildings = 100;

//Generate random coordinates - TODO: Generate value from CfgWorlds to allow compatability with other maps
_generateRandCoords = {
	_randCoords = [(random 12400), (random 12400), 1];
	_randCoords
};

//Determines if camera is the set distance away from nearest building
_awayFromBuilding = {
	params ["_coords", "_distFromBuildings"];
	_nearestBuilding = nearestBuilding _coords;
	_buildingDistActual = _coords distance2D (getPos _nearestBuilding);
	_buildingDistActual > _distFromBuildings;
};

for "_i" from 0 to 600 do 
{
	_generatedCoords = [] call _generateRandCoords;
	_awayFrmBldgBool = [_generatedCoords, _distFromBuildings] call _awayFromBuilding;
	if (_awayFrmBldgBool) then
	{
		//Move camera into position if it's away from buildings
		["Paste", ["Enoch",_generatedCoords,(random 360),0.75,[0.192782,8.41937e-006],0,0,600,(random 1),1,.0.8,0,1]] call BIS_fnc_camera;
		//Sleep before taking screenshot. This allows the LODs to fully pop in and teaxtures to load.
		sleep 5;
		screenshot "";
		sleep 0.5;
	};
};