import React, { useState, useCallback, useEffect } from 'react';
import '../styles/styles.css';

interface InputProps {
	width: number;
	height: number;
	canvasRef: React.RefObject<HTMLCanvasElement>;
	drawingstate: boolean;
	brushsize: number;
	erasersize: number;
	background: boolean;
	realtime: boolean;
	fetchImage: () => void;
}

interface Coordinate {
	x: number;
	y: number;
};

function Canvas({width, height, canvasRef, drawingstate, brushsize, erasersize, background, realtime, fetchImage} : InputProps) {

	// Mouse pointer poisition
	const [mousePosition, setMousePosition] = useState<Coordinate | undefined>(undefined);
	const [isPainting, setIsPainting] = useState(false);

	// 좌표 얻는 함수
	const getCoordinates = (event: MouseEvent): Coordinate | undefined => {
		if (!canvasRef.current) { return; }
		const canvas: HTMLCanvasElement = canvasRef.current;
		const parent: HTMLElement | null = canvas.parentElement;
		if (!parent) { return; }
		return {
			x: event.pageX - parent.offsetLeft,
			y: event.pageY - parent.offsetTop
		};
	};

	// canvas에 선을 긋는 함수
	const drawLine = (originalMousePosition: Coordinate, newMousePosition: Coordinate) => {
		if (!canvasRef.current) { return; }
		const canvas: HTMLCanvasElement = canvasRef.current;
		const context = canvas.getContext('2d');

		if (context) {
			if (drawingstate) {
				context.strokeStyle = "black";  // 선 색깔
				context.lineJoin = 'round';	// 선 끄트머리(?)
				context.lineWidth = brushsize;		// 선 굵기
			} else {
				context.strokeStyle = "white";  // 선 색깔
				context.lineJoin = 'round';	// 선 끄트머리(?)
				context.lineWidth = erasersize;		// 선 굵기
			}

			context.beginPath();
			context.moveTo(originalMousePosition.x, originalMousePosition.y);
			context.lineTo(newMousePosition.x, newMousePosition.y);
			context.closePath();
			context.stroke();
		}
	};

	const startPaint = useCallback((event: MouseEvent) => {
		const coordinates = getCoordinates(event);
		if (coordinates) {
			setIsPainting(true);
			setMousePosition(coordinates);
		}
	}, []);

	const paint = useCallback(
		(event: MouseEvent) => {
			event.preventDefault();   // drag 방지
			event.stopPropagation();  // drag 방지

			if (isPainting) {
				const newMousePosition = getCoordinates(event);
				if (mousePosition && newMousePosition) {
					drawLine(mousePosition, newMousePosition);
					setMousePosition(newMousePosition);
				}
			}
		},
		[isPainting, mousePosition]
	);

	const exitPaint = useCallback(() => {
		setIsPainting(false);
	}, []);

	useEffect(() => {
		if (isPainting) { return; }
		if (realtime === true) { fetchImage(); };
	}, [isPainting]);

	useEffect(() => {
		if (!canvasRef.current) { return; }
		const canvas: HTMLCanvasElement = canvasRef.current;

		const context = canvas.getContext('2d');
		if (context) {
			context.fillStyle = "white";  // 선 색깔
			context.fillRect(0, 0, 512, 512);
		}
	}, []);

	useEffect(() => {
		if (!canvasRef.current) { return; }
		const canvas: HTMLCanvasElement = canvasRef.current;
		canvas.addEventListener('mousedown', startPaint);
		canvas.addEventListener('mousemove', paint);
		canvas.addEventListener('mouseup', exitPaint);
		canvas.addEventListener('mouseleave', exitPaint);

		return () => {
			// Unmount 시 이벤트 리스터 제거
			canvas.removeEventListener('mousedown', startPaint);
			canvas.removeEventListener('mousemove', paint);
			canvas.removeEventListener('mouseup', exitPaint);
			canvas.removeEventListener('mouseleave', exitPaint);
		};
	}, [startPaint, paint, exitPaint]);

	return (
		<div className="canvasBlock">
			<canvas ref={canvasRef} height={height} width={width} className="canvas"/>
			{background ? <div className='background'></div> : null}
		</div>
	);
}

Canvas.defaultProps = {
	width: 512,
	height: 512
};

export default Canvas;