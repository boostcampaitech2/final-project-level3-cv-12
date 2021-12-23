import { useRef, useState } from 'react';
import './App.css';
import Canvas from './canvas/Canvas';
import Output from './output/Output';
import Buttons from './buttons/Buttons';
import Parameters from './buttons/Parameters';
import axios from 'axios';

function App() {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const outputRef = useRef<HTMLCanvasElement>(null);
	const [paramT, setParamT] = useState<number>(0.01);
	const [brushsize, setBrushsize] = useState<number>(4);
	const [erasersize, setErasersize] = useState<number>(10);
	const [drawingstate, setDrawingstate] = useState<boolean>(true);

	const [realtime, setRealtime] = useState<boolean>(true);
	const [loading, setLoading] = useState<boolean>(false);
	const [background, setBackground] = useState<boolean>(true);

	const fetchImage = async () => {
		if (!canvasRef.current) { return; }
		if (!loading) {
			try {
				setLoading(true);
				const data = await canvasRef.current.toDataURL('image/png', 1.0);
				const bodies = await JSON.stringify({"image" : data, "T" : paramT})
				if (!outputRef.current) { return; }
				const canvas: HTMLCanvasElement = outputRef.current;
				const context = await canvas.getContext('2d');
				await axios.post('http://27.96.135.229:8900', bodies, {
					headers: {
						'Content-Type': 'multipart/form-data',
					}})
				if (context) {
					var image = await new Image();
					image.crossOrigin = "Anonymous";
					image.onload = () => {
						if (!context) { return; }
						context.drawImage(image, 0, 0);
					};
					image.src = 'http://27.96.135.229:8900/output.png';
				}
			} catch (e) {
				setLoading(false);
				console.log(e);
				return ;
			};
			setLoading(false);
			return ;
		}
	};

	return (
		<body className="h-screen bg-gradient-to-r from-yellow-400 via-red-500 to-pink-500">
			<div className="App">
				<div className="upside">
					<div className="cvcoin"></div>
					<Buttons canvasRef={canvasRef}
					drawingstate={drawingstate} brushsize={brushsize} erasersize={erasersize} setBrushsize={setBrushsize} setErasersize={setErasersize} setDrawingstate={setDrawingstate}
					realtime={realtime} setRealtime={setRealtime} outputRef={outputRef}
					background={background} loading={loading} setBackground={setBackground} setLoading={setLoading} fetchImage={fetchImage}/>
					<Parameters canvasRef={canvasRef} paramT={paramT} realtime={realtime}
					setParamT={setParamT} outputRef={outputRef} fetchImage={fetchImage}/>
				</div>
				<div className="downside">
					<Canvas canvasRef={canvasRef}
					drawingstate={drawingstate} brushsize={brushsize} erasersize={erasersize} realtime={realtime}
					background={background} fetchImage={fetchImage}/>
					<Output loading={loading} outputRef={outputRef}/>
				</div>
			</div>
		</body>
	);
}

export default App;
