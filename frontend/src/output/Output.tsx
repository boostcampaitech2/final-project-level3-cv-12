import React, { useEffect } from 'react';
import '../styles/styles.css';

interface InputProps {
	width: number;
	height: number;
	loading: boolean;
	outputRef: React.RefObject<HTMLCanvasElement>;
}

function Output({width, height, loading, outputRef} : InputProps) {
	useEffect(() => {
		if (!outputRef.current) { return; }
		const canvas: HTMLCanvasElement = outputRef.current;

		const context = canvas.getContext('2d');
		if (context) {
			context.fillStyle = "white";  // 선 색깔
			context.fillRect(0, 0, 512, 512);
		}
	}, []);

	return (
		<div className="canvasBlock">
			<canvas ref={outputRef} height={height} width={width} className="canvas"/>
			{loading ? <div className='loading'></div> : null}
		</div>
	);
}

Output.defaultProps = {
	width: 512,
	height: 512
};


export default Output;