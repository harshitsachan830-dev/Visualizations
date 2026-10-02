import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

function normalize(value, min, max) {
  return max === min ? 0.5 : (value - min) / (max - min);
}

const pointPalette = ["#4285F4", "#FBBC05", "#34A853", "#EA4335"];
const surfacePalette = ["#EA4335", "#FBBC05", "#34A853", "#4285F4"].map(
  (color) => new THREE.Color(color),
);

export default function DecisionSurface({ surface, points }) {
  const hostRef = useRef(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !surface) return undefined;
    delete host.dataset.renderError;

    let renderer;
    let animationFrame;
    let controls;
    try {
      const width = host.clientWidth;
      const height = host.clientHeight;
      if (!width || !height) return undefined;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#000000");
      const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 60);
      camera.position.set(4.6, 3.9, 5.4);
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        preserveDrawingBuffer: true,
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(width, height);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.15;
      host.appendChild(renderer.domElement);

      scene.add(new THREE.HemisphereLight("#ffffff", "#111111", 2.1));
      const keyLight = new THREE.DirectionalLight("#ffffff", 2.4);
      keyLight.position.set(2.5, 6, 4);
      scene.add(keyLight);

      const xValues = surface.x_values;
      const yValues = surface.y_values;
      const zValues = surface.z_values;
      const rowCount = yValues.length;
      const columnCount = xValues.length;
      const flatZ = zValues.flat();
      const minZ = Math.min(...flatZ);
      const maxZ = Math.max(...flatZ);
      const geometry = new THREE.BufferGeometry();
      const positions = [];
      const colors = [];
      const indices = [];
      const xMin = xValues[0];
      const xMax = xValues.at(-1);
      const yMin = yValues[0];
      const yMax = yValues.at(-1);

      for (let row = 0; row < rowCount; row += 1) {
        for (let column = 0; column < columnCount; column += 1) {
          const value = zValues[row][column];
          const shade = normalize(value, minZ, maxZ);
          positions.push(
            (normalize(xValues[column], xMin, xMax) - 0.5) * 4,
            (shade - 0.5) * 1.7,
            (normalize(yValues[row], yMin, yMax) - 0.5) * 4,
          );
          const colorPosition = shade * (surfacePalette.length - 1);
          const colorIndex = Math.min(Math.floor(colorPosition), surfacePalette.length - 2);
          const color = surfacePalette[colorIndex].clone().lerp(
            surfacePalette[colorIndex + 1],
            colorPosition - colorIndex,
          );
          colors.push(color.r, color.g, color.b);
          const index = row * columnCount + column;
          if (row < rowCount - 1 && column < columnCount - 1) {
            indices.push(
              index,
              index + columnCount,
              index + 1,
              index + 1,
              index + columnCount,
              index + columnCount + 1,
            );
          }
        }
      }
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions, 3),
      );
      geometry.setAttribute(
        "color",
        new THREE.Float32BufferAttribute(colors, 3),
      );
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      const surfaceMesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          roughness: 0.55,
          metalness: 0.04,
        }),
      );
      scene.add(surfaceMesh);

      const grid = new THREE.GridHelper(4, 10, "#FBBC05", "#292929");
      grid.position.y = -0.9;
      scene.add(grid);
      const axes = new THREE.AxesHelper(2.25);
      axes.position.set(-2, -0.88, -2);
      scene.add(axes);

      if (points.length) {
        const pointPositions = [];
        const pointColors = [];
        const labelColors = new Map();
        points.forEach((point) => {
          const xRatio = normalize(point.x, xMin, xMax);
          const yRatio = normalize(point.y, yMin, yMax);
          const gridColumn = Math.max(
            0,
            Math.min(columnCount - 1, Math.round(xRatio * (columnCount - 1))),
          );
          const gridRow = Math.max(
            0,
            Math.min(rowCount - 1, Math.round(yRatio * (rowCount - 1))),
          );
          const level = normalize(zValues[gridRow][gridColumn], minZ, maxZ);
          pointPositions.push(
            (xRatio - 0.5) * 4,
            (level - 0.5) * 1.7 + 0.045,
            (yRatio - 0.5) * 4,
          );
          if (!labelColors.has(point.label))
            labelColors.set(
              point.label,
              new THREE.Color(pointPalette[labelColors.size % pointPalette.length]),
            );
          const color = labelColors.get(point.label);
          pointColors.push(color.r, color.g, color.b);
        });
        const pointsGeometry = new THREE.BufferGeometry();
        pointsGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(pointPositions, 3),
        );
        pointsGeometry.setAttribute(
          "color",
          new THREE.Float32BufferAttribute(pointColors, 3),
        );
        const pointMesh = new THREE.Points(
          pointsGeometry,
          new THREE.PointsMaterial({
            size: 0.075,
            vertexColors: true,
            sizeAttenuation: true,
          }),
        );
        scene.add(pointMesh);
      }

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(0, 0.05, 0);
      controls.minDistance = 4;
      controls.maxDistance = 12;

      const resizeObserver = new ResizeObserver(() => {
        const nextWidth = host.clientWidth;
        const nextHeight = host.clientHeight;
        if (!nextWidth || !nextHeight) return;
        camera.aspect = nextWidth / nextHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(nextWidth, nextHeight);
      });
      resizeObserver.observe(host);

      const render = () => {
        controls.update();
        renderer.render(scene, camera);
        animationFrame = window.requestAnimationFrame(render);
      };
      render();

      return () => {
        window.cancelAnimationFrame(animationFrame);
        resizeObserver.disconnect();
        controls.dispose();
        geometry.dispose();
        surfaceMesh.material.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      };
    } catch {
      host.dataset.renderError = "true";
      renderer?.dispose();
      renderer?.domElement.remove();
      return undefined;
    }
  }, [points, surface]);

  return <div className="surface-viewport" ref={hostRef} />;
}
