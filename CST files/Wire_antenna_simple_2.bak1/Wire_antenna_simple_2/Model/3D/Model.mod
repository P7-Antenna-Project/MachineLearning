'# MWS Version: Version 2021.5 - Jun 28 2021 - ACIS 30.0.1 -

'# length = mm
'# frequency = MHz
'# time = ns
'# frequency range: fmin = 500 fmax = 3000
'# created = '[VERSION]2021.5|30.0.1|20210628[/VERSION]


'@ use template: Antenna - Wire.cfg

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
'set the units
With Units
    .Geometry "mm"
    .Frequency "MHz"
    .Voltage "V"
    .Resistance "Ohm"
    .Inductance "H"
    .TemperatureUnit  "Kelvin"
    .Time "ns"
    .Current "A"
    .Conductance "Siemens"
    .Capacitance "F"
End With

'----------------------------------------------------------------------------

'set the frequency range
Solver.FrequencyRange "500", "3000"

'----------------------------------------------------------------------------

Plot.DrawBox True

With Background
     .Type "Normal"
     .Epsilon "1.0"
     .Mu "1.0"
     .XminSpace "0.0"
     .XmaxSpace "0.0"
     .YminSpace "0.0"
     .YmaxSpace "0.0"
     .ZminSpace "0.0"
     .ZmaxSpace "0.0"
End With

With Boundary
     .Xmin "expanded open"
     .Xmax "expanded open"
     .Ymin "expanded open"
     .Ymax "expanded open"
     .Zmin "expanded open"
     .Zmax "expanded open"
     .Xsymmetry "none"
     .Ysymmetry "none"
     .Zsymmetry "none"
End With

' switch on FD-TET setting for accurate farfields

FDSolver.ExtrudeOpenBC "True"

Mesh.FPBAAvoidNonRegUnite "True"
Mesh.ConsiderSpaceForLowerMeshLimit "False"
Mesh.MinimumStepNumber "5"
Mesh.RatioLimit "20"
Mesh.AutomeshRefineAtPecLines "True", "10"

With MeshSettings
     .SetMeshType "Hex"
     .Set "RatioLimitGeometry", "20"
     .Set "EdgeRefinementOn", "1"
     .Set "EdgeRefinementRatio", "10"
End With

With MeshSettings
     .SetMeshType "Tet"
     .Set "VolMeshGradation", "1.5"
     .Set "SrfMeshGradation", "1.5"
End With

With MeshSettings
     .SetMeshType "HexTLM"
     .Set "RatioLimitGeometry", "20"
End With

PostProcess1D.ActivateOperation "vswr", "true"
PostProcess1D.ActivateOperation "yz-matrices", "true"

With MeshSettings
     .SetMeshType "Srf"
     .Set "Version", 1
End With
IESolver.SetCFIEAlpha "1.000000"

With FarfieldPlot
	.ClearCuts ' lateral=phi, polar=theta
	.AddCut "lateral", "0", "1"
	.AddCut "lateral", "90", "1"
	.AddCut "polar", "90", "1"
End With

'----------------------------------------------------------------------------

With MeshSettings
     .SetMeshType "Hex"
     .Set "Version", 1%
End With

With Mesh
     .MeshType "PBA"
End With

'set the solver type
ChangeSolverType("HF Time Domain")

'----------------------------------------------------------------------------

'@ switch working plane

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Plot.DrawWorkplane "false"

'@ new component: component1

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Component.New "component1"

'@ define brick: component1:Ground_plane

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With Brick
     .Reset 
     .Name "Ground_plane" 
     .Component "component1" 
     .Material "PEC" 
     .Xrange "-ground_length/2", "ground_length/2" 
     .Yrange "-ground_width/2", "ground_width/2" 
     .Zrange "0", "ground_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Pick.PickEndpointFromId "component1:Ground_plane", "4" 
WCS.AlignWCSWithSelectedPoint

'@ align wcs with edge and face

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Pick.PickFaceFromId "component1:Ground_plane", "1" 
Pick.PickEdgeFromId "component1:Ground_plane", "3", "3" 
WCS.AlignWCSWithSelected "EdgeAndFace"

'@ move wcs

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.MoveWCS "local", "0.0", "2", "0.0"

'@ define cylinder: component1:Wire_antenna

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With Cylinder 
     .Reset 
     .Name "Wire_antenna" 
     .Component "component1" 
     .Material "PEC" 
     .OuterRadius "wire_thickness" 
     .InnerRadius "0.0" 
     .Axis "y" 
     .Yrange "0", "wire_length" 
     .Xcenter "0" 
     .Zcenter "wire_height" 
     .Segments "0" 
     .Create 
End With

'@ align wcs with face

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Pick.ForceNextPick 
Pick.PickFaceFromId "component1:Wire_antenna", "1" 
WCS.AlignWCSWithSelected "Face"

'@ move wcs

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.MoveWCS "local", "0.0", "-wire_height-wire_thickness", "0.5"

'@ pick center point

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Pick.PickCenterpointFromId "component1:Wire_antenna", "1"

'@ define discrete port: 1

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With DiscretePort 
     .Reset 
     .PortNumber "1" 
     .Type "SParameter"
     .Label ""
     .Folder ""
     .Impedance "50.0"
     .VoltagePortImpedance "0.0"
     .Voltage "1.0"
     .Current "1.0"
     .Monitor "True"
     .Radius "0.0"
     .SetP1 "True", "0", "2.6", "-0.5"
     .SetP2 "False", "0.0", "0.0", "0.0"
     .InvertDirection "False"
     .LocalCoordinates "True"
     .Wire ""
     .Position "end1"
     .Create 
End With

'@ define time domain solver parameters

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Mesh.SetCreator "High Frequency" 

With Solver 
     .Method "Hexahedral"
     .CalculationType "TD-S"
     .StimulationPort "All"
     .StimulationMode "All"
     .SteadyStateLimit "-40"
     .MeshAdaption "False"
     .AutoNormImpedance "False"
     .NormingImpedance "50"
     .CalculateModesOnly "False"
     .SParaSymmetry "False"
     .StoreTDResultsInCache  "False"
     .FullDeembedding "False"
     .SuperimposePLWExcitation "False"
     .UseSensitivityAnalysis "False"
End With

'@ activate global coordinates

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.ActivateWCS "global"

'@ activate local coordinates

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.ActivateWCS "local"

'@ activate global coordinates

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.ActivateWCS "global"

'@ activate local coordinates

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.ActivateWCS "local"

'@ delete shape: component1:Wire_antenna

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Solid.Delete "component1:Wire_antenna"

'@ delete shape: component1:Ground_plane

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Solid.Delete "component1:Ground_plane"

'@ delete port: port1

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Port.Delete "1"

'@ activate global coordinates

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.ActivateWCS "global"

'@ activate local coordinates

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
WCS.ActivateWCS "local"

'@ set wcs properties

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With WCS
     .SetNormal "-1", "0", "0"
     .SetOrigin "0", "0", "0"
     .SetUVector "0", "-1", "0"
End With

'@ define brick: component1:ground

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With Brick
     .Reset 
     .Name "ground" 
     .Component "component1" 
     .Material "PEC" 
     .Xrange "-ground_width/2", "ground_width/2" 
     .Yrange "0", "ground_length" 
     .Zrange "0", "ground_thickness" 
     .Create
End With

'@ define cylinder: component1:wire_antenna

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With Cylinder 
     .Reset 
     .Name "wire_antenna" 
     .Component "component1" 
     .Material "PEC" 
     .OuterRadius "wire_thickness" 
     .InnerRadius "0.0" 
     .Axis "y" 
     .Yrange "offset_in_v", "offset_in_v+wire_length" 
     .Xcenter "0" 
     .Zcenter "wire_height+ground_thickness" 
     .Segments "0" 
     .Create 
End With

'@ pick end point

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
Pick.PickExtraCirclepointFromId "component1:wire_antenna", "1", "1", "2"

'@ define discrete port: 1

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With DiscretePort 
     .Reset 
     .PortNumber "1" 
     .Type "SParameter"
     .Label ""
     .Folder ""
     .Impedance "50.0"
     .VoltagePortImpedance "0.0"
     .Voltage "1.0"
     .Current "1.0"
     .Monitor "True"
     .Radius "0.0"
     .SetP1 "True", "-1.4695761589768e-16", "3", "21.85"
     .SetP2 "False", "0", "offset_in_v", "ground_thickness"
     .InvertDirection "False"
     .LocalCoordinates "True"
     .Wire ""
     .Position "end1"
     .Create 
End With

'@ define farfield monitor: farfield (f=1900)

'[VERSION]2021.5|30.0.1|20210628[/VERSION]
With Monitor 
     .Reset 
     .Name "farfield (f=1900)" 
     .Domain "Frequency" 
     .FieldType "Farfield" 
     .MonitorValue "1900" 
     .ExportFarfieldSource "False" 
     .UseSubvolume "False" 
     .Coordinates "Structure" 
     .SetSubvolume "-5.55", "0", "-14", "14", "0", "78" 
     .SetSubvolumeOffset "10", "10", "10", "10", "10", "10" 
     .SetSubvolumeInflateWithOffset "False" 
     .SetSubvolumeOffsetType "FractionOfWavelength" 
     .EnableNearfieldCalculation "True" 
     .Create 
End With

