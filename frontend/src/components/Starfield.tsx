import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  r: number;
  base: number;
  twinkleSpeed: number;
  phase: number;
  drift: number;
}

/**
 * Decorative parallax starfield. Purely visual, so it is marked aria-hidden and
 * it stops animating when the user prefers reduced motion.
 */
export function Starfield({ density = 0.00014 }: { density?: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let stars: Star[] = [];
    let frame = 0;
    let width = 0;
    let height = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.round(width * height * density);
      stars = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        r: Math.random() * 1.1 + 0.25,
        base: Math.random() * 0.5 + 0.25,
        twinkleSpeed: Math.random() * 0.02 + 0.004,
        phase: Math.random() * Math.PI * 2,
        drift: Math.random() * 0.05 + 0.008,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      for (const star of stars) {
        const twinkle = reduceMotion
          ? star.base
          : star.base + Math.sin(frame * star.twinkleSpeed + star.phase) * 0.22;
        const alpha = Math.max(0.05, Math.min(1, twinkle));
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
        ctx.fillStyle =
          star.r > 1
            ? `rgba(190, 228, 255, ${alpha})`
            : `rgba(233, 239, 252, ${alpha})`;
        ctx.fill();

        if (!reduceMotion) {
          star.y += star.drift;
          if (star.y > height) {
            star.y = -2;
            star.x = Math.random() * width;
          }
        }
      }
      frame += 1;
    };

    let raf = 0;
    const loop = () => {
      draw();
      raf = window.requestAnimationFrame(loop);
    };

    resize();
    if (reduceMotion) {
      draw();
    } else {
      loop();
    }

    window.addEventListener("resize", resize);
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [density]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      data-testid="starfield"
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  );
}
