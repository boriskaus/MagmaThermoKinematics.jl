# Renders the frames of the 3D showcase movie from the VTK files written by HighRes_Unzen_3D_vtk.jl.
#
# Usage (headless):  pvbatch --force-offscreen-rendering HighRes_Unzen_3D_Paraview.py <dir>/vtk <dir>/frames [scale] [colormap]
# (scale: image size relative to 1600 × 860 pixels; a scale other than 1 renders a figure, without the colour bars and the time;
#  colormap: a text file with one colour per line, r g b in 0–1, for the temperature instead of Inferno)
#
# Left: temperature of the rock below the topography, with a corner cut away; on the vertical faces the layers of the crust (lines of equal
# initial depth: every km in the upper 10 km, every 2 km below), deformed by the sills, dikes and eruptions. Right: the topography, the magma and the sills and dikes (blue, also outlined on the faces on the left).
import os
import sys

from paraview.simple import *

vdir = sys.argv[1] if len(sys.argv) > 1 else "vtk"
outdir = sys.argv[2] if len(sys.argv) > 2 else "frames"
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
cmap = sys.argv[4] if len(sys.argv) > 4 else None
figure = scale != 1.0
os.makedirs(outdir, exist_ok=True)

data = PVDReader(FileName=os.path.join(vdir, "fields.pvd"))
data.PointArrays = ["Temperature", "MeltFraction", "Phases", "InitialDepth"]
surf = PVDReader(FileName=os.path.join(vdir, "surface.pvd"))
times = data.TimestepValues

# rock below the surface, with the corner in front of the volcano cut away
rock = Clip(Input=data, ClipType="Scalar", Scalars=["POINTS", "Phases"], Value=0.5, Invert=0)
cut = Clip(Input=rock, ClipType="Box", Exact=1, Invert=0)                # removed corner: x ≥ 5, y ≤ 4 km (the axis of the volcano)
cut.ClipType.Position = [5.0, -20.0, -21.0]
cut.ClipType.Length = [30.0, 24.0, 30.0]
layers = Contour(Input=ExtractSurface(Input=cut), ContourBy=["POINTS", "InitialDepth"], Isosurfaces=[-float(k) for k in range(1, 11)] + [-12.0, -14.0, -16.0, -18.0])
sill_lines = Contour(Input=ExtractSurface(Input=cut), ContourBy=["POINTS", "Phases"], Isosurfaces=[3.5])   # outlines of the sills and dikes (phase 4) on the faces
sills = Contour(Input=data, ContourBy=["POINTS", "Phases"], Isosurfaces=[3.5])                                # the intrusive complex in 3D
surf_cut = Clip(Input=Transform(Input=surf), ClipType="Box", Exact=1, Invert=0)   # lifted by 50 m so that it does not flicker against the top of the rock
surf_cut.Input.Transform.Translate = [0.0, 0.0, 0.05]
surf_cut.ClipType.Position, surf_cut.ClipType.Length = cut.ClipType.Position, cut.ClipType.Length
melt = Contour(Input=data, ContourBy=["POINTS", "MeltFraction"], Isosurfaces=[0.1], ComputeScalars=1)
melt_hi = Contour(Input=data, ContourBy=["POINTS", "MeltFraction"], Isosurfaces=[0.5], ComputeScalars=1)

layout = CreateLayout("HighRes3D")
left = CreateView("RenderView")
right = CreateView("RenderView")
layout.SplitHorizontal(0, 0.5)
layout.AssignView(1, left)
layout.AssignView(2, right)
layout.SetSize(1600, 860)

T_lut = GetColorTransferFunction("Temperature")
if cmap is None:
    T_lut.ApplyPreset("Inferno", True)
else:                                                                          # equally spaced colours over 0–1 (rescaled to 0–1000 °C below)
    rgb = [[float(v) for v in line.split()] for line in open(cmap) if line.strip()]
    T_lut.RGBPoints = [c for i, col in enumerate(rgb) for c in [i / (len(rgb) - 1)] + col]
Z_lut = GetColorTransferFunction("Elevation")                                  # topographic colours [km]: blue sea, a jump at sea level, green to brown to white
Z_lut.RGBPoints = [-0.5, 0.05, 0.20, 0.45, -0.001, 0.50, 0.75, 0.92, 0.0, 0.20, 0.50, 0.22, 0.3, 0.45, 0.66, 0.30,
                   0.8, 0.82, 0.76, 0.48, 1.2, 0.60, 0.44, 0.30, 1.6, 0.97, 0.97, 0.97]
Z_lut.AutomaticRescaleRangeMode = "Never"
SILL_COLOR = [0.0, 0.75, 1.0]                                                  # sills and dikes (deep sky blue, as in 2D)
M_lut = GetColorTransferFunction("MeltFraction")                               # pale yellow (little melt) to orange to deep red (magma)
M_lut.RGBPoints = [0.0, 1.0, 0.95, 0.6, 0.3, 1.0, 0.6, 0.1, 0.6, 0.9, 0.2, 0.05, 1.0, 0.55, 0.0, 0.05]

d = Show(cut, left)
ColorBy(d, ("POINTS", "Temperature"))
d.LookupTable = T_lut
d = Show(layers, left)
ColorBy(d, None)
d.AmbientColor = d.DiffuseColor = [1.0, 1.0, 1.0]
d.Opacity = 0.12                                                               # faint, so the magma inside the complex shows
d.LineWidth = 1.5
d = Show(sill_lines, left)
ColorBy(d, None)
d.AmbientColor = d.DiffuseColor = SILL_COLOR
d.LineWidth = 2.0
d = Show(surf_cut, left)
ColorBy(d, ("POINTS", "Elevation"))
d.LookupTable = Z_lut
Show(Outline(Input=data), left).DiffuseColor = [0.5, 0.5, 0.5]

d = Show(surf, right)
ColorBy(d, ("POINTS", "Elevation"))
d.LookupTable = Z_lut
d.Opacity = 0.4
d = Show(sills, right)
ColorBy(d, None)
d.AmbientColor = d.DiffuseColor = SILL_COLOR
d.Opacity = 0.12                                                               # faint, so the magma inside the complex shows
for m, opacity in ((melt, 0.4), (melt_hi, 1.0)):
    d = Show(m, right)
    ColorBy(d, ("POINTS", "MeltFraction"))
    d.LookupTable = M_lut
    d.Opacity = opacity
    d.Ambient = 0.35                                                          # lit from all sides, so the magma stays bright below the surface
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
    bar.Visibility = 0 if figure else 1
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraPosition = [80.0, -75.0, 40.0]
    view.CameraFocalPoint = [2.0, 2.0, -9.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraViewAngle = 30.0

for lut, lo, hi in ((T_lut, 0.0, 1000.0), (M_lut, 0.0, 1.0)):
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(lo, hi)

label = Text(Text="")
t = Show(label, right)
t.WindowLocation = "Upper Right Corner"
t.Color = [0.1, 0.1, 0.1]
t.FontSize = 18
t.Visibility = 0 if figure else 1

for i, time in enumerate(times):
    label.Text = "%.1f kyr before present" % abs(time)
    for view in (left, right):
        view.ViewTime = time
    SaveScreenshot(os.path.join(outdir, "frame_%04d.png" % i), layout, ImageResolution=[round(1600 * scale), round(860 * scale)])
