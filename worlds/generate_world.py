"""Writes worlds/arena.wbt from the SAME object list the offline simulator uses (navlib/world.py).
Run:  python worlds/generate_world.py     then open worlds/arena.wbt in Webots (R2023b or newer)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from navlib import config as C
from navlib.world import default_objects

RGB = {"orange": (1.0, 0.5, 0.0), "blue": (0.0, 0.0, 1.0), "green": (0.0, 0.78, 0.0),
       "magenta": (1.0, 0.0, 1.0), "gray": (0.37, 0.37, 0.37)}


def appearance(color):
    r, g, b = RGB[color]
    return f"PBRAppearance {{ baseColor {r} {g} {b} roughness 1 metalness 0 }}"


def solid(o):
    if o["kind"] == "box":
        geom = f"Box {{ size {o['sx']} {o['sy']} {o['h']} }}"
    else:
        geom = f"Cylinder {{ height {o['h']} radius {o['r']} }}"
    return f"""Solid {{
  translation {o['cx']} {o['cy']} {o['h'] / 2}
  name "{o['name']}"
  children [ Shape {{ appearance {appearance(o['color'])} geometry {geom} }} ]
  boundingObject {geom}
}}"""


def robot():
    sx, sy = C.START
    return f"""Robot {{
  translation {sx} {sy} {C.WHEEL_R}
  rotation 0 0 1 1.5707963
  name "nav_robot"
  controller "nav_controller"
  children [
    Shape {{ appearance PBRAppearance {{ baseColor 0.1 0.3 0.8 roughness 0.6 metalness 0 }}
            geometry Box {{ size 0.30 0.20 0.08 }} }}
    GPS {{ name "gps" }}
    InertialUnit {{ name "imu" }}
    Lidar {{
      name "lidar"
      translation 0 0 {C.LIDAR_HEIGHT - C.WHEEL_R}
      horizontalResolution {C.LIDAR_BEAMS}
      fieldOfView 6.2831853
      numberOfLayers 1
      minRange {C.LIDAR_RMIN}
      maxRange {C.LIDAR_RMAX}
    }}
    Camera {{
      name "camera"
      translation {C.CAM_XOFF} 0 {C.CAM_HEIGHT - C.WHEEL_R}
      width {C.CAM_W}
      height {C.CAM_H}
      fieldOfView {C.CAM_FOV}
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 1 0 anchor 0 {C.TRACK / 2} 0 }}
      device [ RotationalMotor {{ name "left wheel motor" maxVelocity 15 }}
               PositionSensor {{ name "left wheel sensor" }} ]
      endPoint Solid {{
        translation 0 {C.TRACK / 2} 0
        rotation 1 0 0 1.5707963
        children [ Shape {{ appearance PBRAppearance {{ baseColor 0.05 0.05 0.05 roughness 1 metalness 0 }}
                           geometry Cylinder {{ height 0.02 radius {C.WHEEL_R} }} }} ]
        name "left wheel"
        boundingObject Cylinder {{ height 0.02 radius {C.WHEEL_R} }}
        physics Physics {{ density -1 mass 0.2 }}
      }}
    }}
    HingeJoint {{
      jointParameters HingeJointParameters {{ axis 0 1 0 anchor 0 {-C.TRACK / 2} 0 }}
      device [ RotationalMotor {{ name "right wheel motor" maxVelocity 15 }}
               PositionSensor {{ name "right wheel sensor" }} ]
      endPoint Solid {{
        translation 0 {-C.TRACK / 2} 0
        rotation 1 0 0 1.5707963
        children [ Shape {{ appearance PBRAppearance {{ baseColor 0.05 0.05 0.05 roughness 1 metalness 0 }}
                           geometry Cylinder {{ height 0.02 radius {C.WHEEL_R} }} }} ]
        name "right wheel"
        boundingObject Cylinder {{ height 0.02 radius {C.WHEEL_R} }}
        physics Physics {{ density -1 mass 0.2 }}
      }}
    }}
    Solid {{
      translation -0.12 0 -0.03
      name "caster"
      contactMaterial "caster"
      children [ Shape {{ appearance PBRAppearance {{ baseColor 0.6 0.6 0.6 roughness 1 metalness 0 }}
                         geometry Sphere {{ radius 0.02 }} }} ]
      boundingObject Sphere {{ radius 0.02 }}
      physics Physics {{ density -1 mass 0.05 }}
    }}
  ]
  boundingObject Box {{ size 0.30 0.20 0.08 }}
  physics Physics {{ density -1 mass 2.0 }}
}}"""


def main():
    A = C.ARENA
    parts = ["""#VRML_SIM R2023b utf8

WorldInfo {
  basicTimeStep 32
  coordinateSystem "ENU"
  contactProperties [ ContactProperties { material1 "caster" material2 "default" coulombFriction [ 0 ] } ]
}
Viewpoint {
  orientation 0 0.95 -0.3 4.71
  position 5 -6 13
}
Background { skyColor [ 0.85 0.9 1 ] }
DirectionalLight { direction 0.3 0.4 -1 intensity 1.2 }
DirectionalLight { direction -0.4 -0.3 -1 intensity 0.6 }
Solid {
  translation %g %g -0.005
  name "floor"
  children [ Shape { appearance PBRAppearance { baseColor 0.78 0.78 0.78 roughness 1 metalness 0 } geometry Box { size %g %g 0.01 } } ]
  boundingObject Box { size %g %g 0.01 }
}""" % (A / 2, A / 2, A + 0.4, A + 0.4, A + 0.4, A + 0.4)]
    parts += [solid(o) for o in default_objects()]
    parts.append(robot())
    out = os.path.join(os.path.dirname(__file__), "arena.wbt")
    open(out, "w").write("\n".join(parts) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
