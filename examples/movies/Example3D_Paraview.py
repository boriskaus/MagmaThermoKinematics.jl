# Renders the frames of the movie Example3D.gif from the output of examples/Example3D.jl.
#
# Usage (headless, from the directory in which Example3D.jl was run):
#   pvbatch --force-offscreen-rendering Example3D_Paraview.py Example3D.pvd frames/
#
# Left: temperature of the rock below the topography, with a corner cut
# away. Right: the topography (colored by elevation) and the molten region below it.
import os
import sys

from paraview.simple import *

pvd = sys.argv[1] if len(sys.argv) > 1 else "Example3D.pvd"
outdir = sys.argv[2] if len(sys.argv) > 2 else "frames"
os.makedirs(outdir, exist_ok=True)

data = PVDReader(FileName=pvd)
data.PointArrays = ["Temperature", "MeltFraction", "Phases"]
times = data.TimestepValues

# rock (Phases = 1) below the surface, with a corner cut away so we can look inside
rock = Clip(Input=data, ClipType="Scalar", Scalars=["POINTS", "Phases"], Value=0.9, Invert=0)
cut = Clip(Input=rock, ClipType="Box", Exact=1, Invert=0)
cut.ClipType.Position = [0.0, -16.0, -14.0]
cut.ClipType.Length = [16.0, 16.0, 20.0]
hot = Contour(Input=cut, ContourBy=["POINTS", "Temperature"], Isosurfaces=[700.0])

# topography (Phases = 0.5) colored by elevation, and the molten region
surf = Contour(Input=data, ContourBy=["POINTS", "Phases"], Isosurfaces=[0.5], ComputeNormals=0)
smooth = Smooth(Input=surf, NumberofIterations=300)  # removes the terraces of the grid
elev = Calculator(Input=SurfaceNormals(Input=smooth), ResultArrayName="Elevation", Function="coordsZ")
elev_cut = Clip(Input=elev, ClipType="Box", Exact=1, Invert=0)
elev_cut.ClipType.Position, elev_cut.ClipType.Length = cut.ClipType.Position, cut.ClipType.Length
melt = Contour(Input=data, ContourBy=["POINTS", "MeltFraction"], Isosurfaces=[0.1], ComputeScalars=1)
melt_hi = Contour(Input=data, ContourBy=["POINTS", "MeltFraction"], Isosurfaces=[0.5], ComputeScalars=1)

layout = CreateLayout("Example3D")
left = CreateView("RenderView")
right = CreateView("RenderView")
layout.SplitHorizontal(0, 0.5)
layout.AssignView(1, left)
layout.AssignView(2, right)
layout.SetSize(1400, 840)

T_lut = GetColorTransferFunction("Temperature")
T_lut.ApplyPreset("Inferno", True)
T_lut.AutomaticRescaleRangeMode = "Never"
T_lut.RescaleTransferFunction(0.0, 900.0)
Z_lut = GetColorTransferFunction("Elevation")
Z_lut.ApplyPreset("gist_earth", True)
Z_lut.AutomaticRescaleRangeMode = "Never"
Z_lut.RescaleTransferFunction(0.0, 4.0)
M_lut = GetColorTransferFunction("MeltFraction")
M_lut.ApplyPreset("Black-Body Radiation", True)
M_lut.InvertTransferFunction()
M_lut.AutomaticRescaleRangeMode = "Never"
M_lut.RescaleTransferFunction(0.0, 1.0)

d = Show(cut, left)
ColorBy(d, ("POINTS", "Temperature"))
d.LookupTable = T_lut
d = Show(hot, left)
ColorBy(d, None)
d.DiffuseColor = [1.0, 0.93, 0.7]
d = Show(elev_cut, left)
ColorBy(d, ("POINTS", "Elevation"))
d.LookupTable = Z_lut
Show(Outline(Input=data), left).DiffuseColor = [0.5, 0.5, 0.5]

d = Show(elev, right)
ColorBy(d, ("POINTS", "Elevation"))
d.LookupTable = Z_lut
for m, opacity in ((melt, 0.5), (melt_hi, 1.0)):
    d = Show(m, right)
    ColorBy(d, ("POINTS", "MeltFraction"))
    d.LookupTable = M_lut
    d.Opacity = opacity
Show(Outline(Input=data), right).DiffuseColor = [0.5, 0.5, 0.5]

for view, lut, title, fmt in ((left, T_lut, "Temperature [°C]", "%.0f"), (right, M_lut, "Melt fraction", "%.1f")):
    bar = GetScalarBar(lut, view)
    bar.Title, bar.ComponentTitle = title, ""
    bar.TitleFontSize, bar.LabelFontSize = 16, 14
    bar.RangeLabelFormat = bar.LabelFormat = fmt
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Lower Center"
    bar.TitleColor = bar.LabelColor = [0.1, 0.1, 0.1]
    bar.ScalarBarLength = 0.6
    bar.Visibility = 1
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraPosition = [72.0, -86.0, 34.0]
    view.CameraFocalPoint = [0.0, 0.0, -13.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraViewAngle = 30.0

for lut, lim in ((T_lut, 900.0), (Z_lut, 4.0), (M_lut, 1.0)):
    lut.RescaleTransferFunction(0.0, lim)  # Show/ColorBy rescale to the data range

label = Text(Text="")
t = Show(label, right)
t.WindowLocation = "Upper Right Corner"
t.Color = [0.1, 0.1, 0.1]
t.FontSize = 18

for i, time in enumerate(times):
    label.Text = "%.2f kyrs" % time
    for view in (left, right):
        view.ViewTime = time
    SaveScreenshot(os.path.join(outdir, "frame_%04d.png" % i), layout, ImageResolution=[1400, 840])
