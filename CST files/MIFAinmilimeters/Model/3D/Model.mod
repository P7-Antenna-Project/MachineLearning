'# MWS Version: Version 2023.5 - Jun 08 2023 - ACIS 32.0.1 -

'# length = mm
'# frequency = MHz
'# time = ns
'# frequency range: fmin = 500 fmax = 3000
'# created = '[VERSION]2023.5|32.0.1|20230608[/VERSION]


'@ use template: Antenna - Mobile Device, sub 6 GHz.cfg

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
'set the units
With Units
    .Geometry "mm"
    .Frequency "MHz"
    .Voltage "V"
    .Resistance "Ohm"
    .Inductance "H"
    .TemperatureUnit  "Celsius"
    .Time "ns"
    .Current "A"
    .Conductance "Siemens"
    .Capacitance "F"
End With

ThermalSolver.AmbientTemperature "0"

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

' optimize mesh settings for planar structures

With Mesh
     .MergeThinPECLayerFixpoints "True"
     .RatioLimit "20"
     .FPBAAvoidNonRegUnite "True"
     .ConsiderSpaceForLowerMeshLimit "False"
     .MinimumStepNumber "5"
     .AutoMeshNumberOfShapeFaces "300"
     .SetGenericUserFlag("AllowPowerLossPP", True)
End With

With MeshSettings
     .SetMeshType "Hex"
     .Set "RatioLimitGeometry", "20"
End With

With MeshSettings
     .SetMeshType "HexTLM"
     .Set "RatioLimitGeometry", "20"
End With

' change mesh adaption scheme to energy
' 		(planar structures tend to store high energy
'     	 locally at edges rather than globally in volume)

MeshAdaption3D.SetAdaptionStrategy "Energy"

' switch on FD-TET setting for accurate farfields

FDSolver.ExtrudeOpenBC "True"

Solver.PrepareFarfields "False"

PostProcess1D.ActivateOperation "vswr", "true"
PostProcess1D.ActivateOperation "yz-matrices", "true"

With FarfieldPlot
	.ClearCuts ' lateral=phi, polar=theta
	.AddCut "lateral", "0", "1"
	.AddCut "lateral", "90", "1"
	.AddCut "polar", "90", "1"
End With

'----------------------------------------------------------------------------

Dim sDefineAt As String
sDefineAt = "500;1750;3000"
Dim sDefineAtName As String
sDefineAtName = "500;1750;3000"
Dim sDefineAtToken As String
sDefineAtToken = "f="
Dim aFreq() As String
aFreq = Split(sDefineAt, ";")
Dim aNames() As String
aNames = Split(sDefineAtName, ";")

Dim nIndex As Integer
For nIndex = LBound(aFreq) To UBound(aFreq)

Dim zz_val As String
zz_val = aFreq (nIndex)
Dim zz_name As String
zz_name = sDefineAtToken & aNames (nIndex)

' Define Farfield Monitors
With Monitor
    .Reset
    .Name "farfield ("& zz_name &")"
    .Domain "Frequency"
    .FieldType "Farfield"
    .MonitorValue  zz_val
    .ExportFarfieldSource "False"
    .Create
End With

Next

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

'@ define material: Folder1/FR-4

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Material
     .Reset
     .Name "FR-4"
     .Folder "Folder1"
     .Rho "0.0"
     .ThermalType "Normal"
     .ThermalConductivity "0.3"
     .SpecificHeat "0", "J/K/kg"
     .DynamicViscosity "0"
     .Emissivity "0"
     .MetabolicRate "0.0"
     .VoxelConvection "0.0"
     .BloodFlow "0"
     .MechanicsType "Unused"
     .FrqType "all"
     .Type "Normal"
     .MaterialUnit "Frequency", "MHz"
     .MaterialUnit "Geometry", "mm"
     .MaterialUnit "Time", "ns"
     .MaterialUnit "Temperature", "Celsius"
     .Epsilon "4.3"
     .Mu "1"
     .Sigma "0.0"
     .TanD "0.025"
     .TanDFreq "0.0"
     .TanDGiven "True"
     .TanDModel "ConstTanD"
     .SetConstTanDStrategyEps "AutomaticOrder"
     .ConstTanDModelOrderEps "3"
     .DjordjevicSarkarUpperFreqEps "0"
     .SetElParametricConductivity "False"
     .ReferenceCoordSystem "Global"
     .CoordSystemType "Cartesian"
     .SigmaM "0"
     .TanDM "0.0"
     .TanDMFreq "0.0"
     .TanDMGiven "False"
     .TanDMModel "ConstTanD"
     .SetConstTanDStrategyMu "AutomaticOrder"
     .ConstTanDModelOrderMu "3"
     .DjordjevicSarkarUpperFreqMu "0"
     .SetMagParametricConductivity "False"
     .DispModelEps "None"
     .DispModelMu "None"
     .DispersiveFittingSchemeEps "Nth Order"
     .MaximalOrderNthModelFitEps "10"
     .ErrorLimitNthModelFitEps "0.1"
     .DispersiveFittingSchemeMu "Nth Order"
     .MaximalOrderNthModelFitMu "10"
     .ErrorLimitNthModelFitMu "0.1"
     .UseGeneralDispersionEps "False"
     .UseGeneralDispersionMu "False"
     .NLAnisotropy "False"
     .NLAStackingFactor "1"
     .NLADirectionX "1"
     .NLADirectionY "0"
     .NLADirectionZ "0"
     .Colour "1", "0.501961", "0.25098" 
     .Wireframe "False" 
     .Reflection "False" 
     .Allowoutline "True" 
     .Transparentoutline "False" 
     .Transparency "0" 
     .Create
End With

'@ new component: component1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Component.New "component1"

'@ define brick: component1:substrate

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "substrate" 
     .Component "component1" 
     .Material "Folder1/FR-4" 
     .Xrange "substrateX_min", "substrateX_max" 
     .Yrange "substrateY_min", "substrateY_max" 
     .Zrange "0", "substrateH" 
     .Create
End With

'@ new component: antenna

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Component.New "antenna"

'@ define brick: antenna:Feed

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Feed" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "stripWidth" 
     .Yrange "-feed_h_negative", "feed_h" 
     .Zrange "substrateH", "substrateH+copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Feed", "8" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:top1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "top1" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "Tw1" 
     .Yrange "0", "stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:top1", "6" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:line1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "line1" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "-stripWidth" 
     .Yrange "0", "-Line1_height" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:line1", "7" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Bot1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Bot1" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "Bw1" 
     .Yrange "0", "-stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Bot1", "5" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Line2

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Line2" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "-stripWidth" 
     .Yrange "0", "Line2_height" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Line2", "8" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Top2

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Top2" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "Tw2" 
     .Yrange "0", "stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Top2", "6" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Line3

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Line3" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "-stripWidth" 
     .Yrange "0", "-line3_height" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ rename block: antenna:line1 to: antenna:Line1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Rename "antenna:line1", "Line1"

'@ rename block: antenna:top1 to: antenna:Top1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Rename "antenna:top1", "Top1"

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Line3", "7" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Bot2

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Bot2" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "Bw2" 
     .Yrange "0", "-stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Bot2", "5" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Line4

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Line4" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "-stripWidth" 
     .Yrange "0", "Line4_height" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Line4", "8" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Top3

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Top3" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "tw3" 
     .Yrange "0", "stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Top3", "6" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Line5

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Line5" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "-stripWidth" 
     .Yrange "0", "-Line5_height" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Line5", "7" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Bot3

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Bot3" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "Bw3" 
     .Yrange "0", "-stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Bot3", "5" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Line6

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Line6" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "-stripWidth" 
     .Yrange "0", "Line6_height" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Line6", "8" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:Tip

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Tip" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "tipLength" 
     .Yrange "0", "stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Feed", "6" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: component1:Leg

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Leg" 
     .Component "component1" 
     .Material "Folder1/FR-4" 
     .Xrange "0", "Leg_length" 
     .Yrange "Leg_placement", "Leg_placement+stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ change material and color: component1:Leg to: PEC

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.ChangeMaterial "component1:Leg", "PEC" 
Solid.SetUseIndividualColor "component1:Leg", 1
Solid.ChangeIndividualColor "component1:Leg", "128", "128", "128"

'@ delete shape: component1:Leg

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Delete "component1:Leg"

'@ define brick: antenna:Leg

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "Leg" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "Leg_length" 
     .Yrange "Leg_placement", "Leg_placement+stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ clear picks

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.ClearAllPicks

'@ activate global coordinates

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
WCS.ActivateWCS "global"

'@ define brick: antenna:GP

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "GP" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "gp_x_min", "gp_x_max" 
     .Yrange "gp_y_min", "gp_y_max" 
     .Zrange "0", "substrateH+copper_thickness" 
     .Create
End With

'@ define brick: component1:GPsubstrate

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "GPsubstrate" 
     .Component "component1" 
     .Material "Folder1/FR-4" 
     .Xrange "-gp_x_max", "gp_x_max" 
     .Yrange "-gp_x_max", "gp_x_max" 
     .Zrange "0", "substrateH" 
     .Create
End With

'@ delete shape: component1:GPsubstrate

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Delete "component1:GPsubstrate"

'@ define brick: component1:GPsubstrate

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "GPsubstrate" 
     .Component "component1" 
     .Material "Folder1/FR-4" 
     .Xrange "-gp_x_min", "gp_x_max" 
     .Yrange "-gp_y_min", "gp_y_max" 
     .Zrange "0", "substrateH" 
     .Create
End With

'@ delete shape: component1:GPsubstrate

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Delete "component1:GPsubstrate"

'@ clear picks

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.ClearAllPicks

'@ pick center point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickCenterpointFromId "antenna:Feed", "3"

'@ define brick: component1:hole_in_gp

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "hole_in_gp" 
     .Component "component1" 
     .Material "PEC" 
     .Xrange "-20/2", "20/2" 
     .Yrange "0", "-holeL" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ delete shape: component1:hole_in_gp

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Delete "component1:hole_in_gp"

'@ clear picks

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.ClearAllPicks

'@ define brick: component1:hole_in_GP

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "hole_in_GP" 
     .Component "component1" 
     .Material "Folder1/FR-4" 
     .Xrange "0", "20" 
     .Yrange "0", "holeL" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ delete shape: component1:hole_in_GP

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Delete "component1:hole_in_GP"

'@ define brick: component1:HoleInGp

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "HoleInGp" 
     .Component "component1" 
     .Material "PEC" 
     .Xrange "-holeW", "holeWPlus" 
     .Yrange "0", "-holeL" 
     .Zrange "0", "substrateH+copper_thickness" 
     .Create
End With

'@ boolean subtract shapes: antenna:GP, component1:HoleInGp

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Subtract "antenna:GP", "component1:HoleInGp"

'@ pick center point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickCenterpointFromId "antenna:Feed", "3"

'@ define discrete port: 1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With DiscretePort 
     .Reset 
     .PortNumber "1" 
     .Type "SParameter"
     .Label ""
     .Folder ""
     .Impedance "50.0"
     .Voltage "1.0"
     .Current "1.0"
     .Monitor "True"
     .Radius "0.0"
     .SetP1 "True", "10", "0", "5.5"
     .SetP2 "False", "10", "-holeL", "5.5"
     .InvertDirection "False"
     .LocalCoordinates "False"
     .Wire ""
     .Position "end1"
     .Create 
End With

'@ define time domain solver parameters

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
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
     .RunDiscretizerOnly "False"
     .FullDeembedding "False"
     .SuperimposePLWExcitation "False"
     .UseSensitivityAnalysis "False"
End With

'@ set PBA version

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Discretizer.PBAVersion "2023060823"

'@ define units

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Units 
     .SetUnit "Length", "mil"
     .SetUnit "Temperature", "degC"
     .SetUnit "Voltage", "V"
     .SetUnit "Current", "A"
     .SetUnit "Resistance", "Ohm"
     .SetUnit "Conductance", "S"
     .SetUnit "Capacitance", "F"
     .SetUnit "Inductance", "H"
     .SetUnit "Frequency", "MHz"
     .SetUnit "Time", "ns"
     .SetResultUnit "frequency", "frequency", "" 
End With

'@ delete port: port1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Port.Delete "1"

'@ pick center point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickCenterpointFromId "antenna:Feed", "3"

'@ pick mid point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickMidpointFromId "antenna:GP", "25"

'@ define discrete port: 1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With DiscretePort 
     .Reset 
     .PortNumber "1" 
     .Type "SParameter"
     .Label ""
     .Folder ""
     .Impedance "50.0"
     .Voltage "1.0"
     .Current "1.0"
     .Monitor "True"
     .Radius "0.0"
     .SetP1 "True", "10", "0", "3.15"
     .SetP2 "True", "10", "-10", "3.3"
     .InvertDirection "False"
     .LocalCoordinates "False"
     .Wire ""
     .Position "end1"
     .Create 
End With

'@ delete shape: antenna:GP

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.Delete "antenna:GP"

'@ define brick: component1:GP

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "GP" 
     .Component "component1" 
     .Material "Folder1/FR-4" 
     .Xrange "gp_x_min", "gp_x_max" 
     .Yrange "gp_y_min", "gp_y_max" 
     .Zrange "0", "-copper_thickness" 
     .Create
End With

'@ change material and color: component1:GP to: PEC

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Solid.ChangeMaterial "component1:GP", "PEC" 
Solid.SetUseIndividualColor "component1:GP", 1
Solid.ChangeIndividualColor "component1:GP", "128", "128", "128"

'@ delete port: port1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Port.Delete "1"

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:Top1", "8" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:groundingPinTop

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "groundingPinTop" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "-groundingPinTopLength" 
     .Yrange "0", "-stripWidth" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:groundingPinTop", "7" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:groundingPinLength

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "groundingPinLength" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "stripWidth" 
     .Yrange "0", "-groundingPinLength" 
     .Zrange "0", "copper_thickness" 
     .Create
End With

'@ align wcs with point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickEndpointFromId "antenna:groundingPinLength", "7" 
WCS.AlignWCSWithSelectedPoint

'@ define brick: antenna:gp_gpConnection

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Brick
     .Reset 
     .Name "gp_gpConnection" 
     .Component "antenna" 
     .Material "PEC" 
     .Xrange "0", "stripWidth" 
     .Yrange "0", "copper_thickness" 
     .Zrange "0", "-substrateH-(copper_thickness/2)" 
     .Create
End With

'@ pick center point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickCenterpointFromId "antenna:Feed", "3"

'@ pick mid point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickMidpointFromId "component1:GP", "4"

'@ define discrete port: 1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With DiscretePort 
     .Reset 
     .PortNumber "1" 
     .Type "SParameter"
     .Label ""
     .Folder ""
     .Impedance "50.0"
     .Voltage "1.0"
     .Current "1.0"
     .Monitor "True"
     .Radius "0.0"
     .SetP1 "True", "75", "25", "5"
     .SetP2 "True", "215", "-275", "-3"
     .InvertDirection "False"
     .LocalCoordinates "True"
     .Wire ""
     .Position "end1"
     .Create 
End With

'@ define units

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Units 
     .SetUnit "Length", "mm"
     .SetUnit "Temperature", "degC"
     .SetUnit "Voltage", "V"
     .SetUnit "Current", "A"
     .SetUnit "Resistance", "Ohm"
     .SetUnit "Conductance", "S"
     .SetUnit "Capacitance", "F"
     .SetUnit "Inductance", "H"
     .SetUnit "Frequency", "MHz"
     .SetUnit "Time", "ns"
     .SetResultUnit "frequency", "frequency", "" 
End With

'@ create group: meshgroup1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Group.Add "meshgroup1", "mesh"

'@ add items to group: "meshgroup1"

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Group.AddItem "solid$antenna:Leg", "meshgroup1"

'@ add items to group: "Excluded from Simulation"

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Group.AddItem "solid$antenna:Leg", "Excluded from Simulation"

'@ add items to group: "Excluded from Bounding Box"

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Group.AddItem "solid$antenna:Leg", "Excluded from Bounding Box"

'@ delete port: port1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Port.Delete "1"

'@ pick center point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickCenterpointFromId "antenna:Feed", "3"

'@ pick center point

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Pick.PickCenterpointFromId "component1:GP", "1"

'@ define discrete port: 1

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With DiscretePort 
     .Reset 
     .PortNumber "1" 
     .Type "SParameter"
     .Label ""
     .Folder ""
     .Impedance "50.0"
     .Voltage "1.0"
     .Current "1.0"
     .Monitor "True"
     .Radius "0.0"
     .SetP1 "True", "2.35", "0.827", "0.05"
     .SetP2 "True", "7", "-2.983", "-1"
     .InvertDirection "False"
     .LocalCoordinates "True"
     .Wire ""
     .Position "end1"
     .Create 
End With

'@ define farfield monitor: farfield (f=1900)

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
With Monitor 
     .Reset 
     .Name "farfield (f=1900)" 
     .Domain "Frequency" 
     .FieldType "Farfield" 
     .MonitorValue "1900" 
     .ExportFarfieldSource "False" 
     .UseSubvolume "False" 
     .Coordinates "Structure" 
     .SetSubvolume "-10", "35.2", "-20", "7.62", "-0.1", "1.1" 
     .SetSubvolumeOffset "10", "10", "10", "10", "10", "10" 
     .SetSubvolumeInflateWithOffset "False" 
     .SetSubvolumeOffsetType "FractionOfWavelength" 
     .EnableNearfieldCalculation "True" 
     .Create 
End With

'@ delete monitors

'[VERSION]2023.5|32.0.1|20230608[/VERSION]
Monitor.Delete "farfield (f=1750)" 
Monitor.Delete "farfield (f=3000)" 
Monitor.Delete "farfield (f=500)"

