import '../styles/styles.css';

interface InputProps {
	canvasRef: React.RefObject<HTMLCanvasElement>;
	drawingstate: boolean;
	brushsize: number;
	erasersize: number;
	setBrushsize: React.Dispatch<React.SetStateAction<number>>;
	setErasersize: React.Dispatch<React.SetStateAction<number>>;
	setDrawingstate: React.Dispatch<React.SetStateAction<boolean>>;
	background: boolean;
	loading: boolean;
	setBackground: React.Dispatch<React.SetStateAction<boolean>>;
	setLoading: React.Dispatch<React.SetStateAction<boolean>>;
	fetchImage: () => void;
	outputRef: React.RefObject<HTMLCanvasElement>;
	realtime: boolean;
	setRealtime: React.Dispatch<React.SetStateAction<boolean>>;
}


function Buttons({canvasRef, drawingstate, brushsize, erasersize, setBrushsize, setErasersize,
	setDrawingstate, outputRef, background, setBackground, realtime, setRealtime, fetchImage} : InputProps) {

	const convertCanvases = () => {
		if (!canvasRef.current) { return; }
		fetchImage();
	}

	const handleBackground = (event: React.ChangeEvent<HTMLInputElement>) : number | undefined => {
		if (!canvasRef.current) { return; }
		setBackground(!background);
	}

	const handleRealtime = () => {
		if (!canvasRef.current) { return; }
		setRealtime(!realtime);
	}

	const handleBrush = (event: React.ChangeEvent<HTMLInputElement>) : number | undefined => {
		if (!canvasRef.current) { return; }
		const size:string = event.target.value;
		setBrushsize(+size);
	}
	const handleEraser = (event: React.ChangeEvent<HTMLInputElement>) : number | undefined => {
		if (!canvasRef.current) { return; }
		const size:string = event.target.value;
		setErasersize(+size);
	}
	const modeChange = () => {
		if (!canvasRef.current) { return; }
		setDrawingstate(!drawingstate);
	}
	const clearCanvas = () => {
		if (!canvasRef.current) { return; }
		const canvas: HTMLCanvasElement = canvasRef.current;
		const context = canvas.getContext('2d');
		if (context) {
			context.fillStyle = "white";  // 선 색깔
			context.fillRect(0, 0, 512, 512);
		}
	}

	const downloadhandler = function(){
		if (!outputRef.current) { return; }
		const canvas: HTMLCanvasElement = outputRef.current;
		const context = canvas.getContext('2d');
		const link = document.createElement('a');
		if (context) {
			link.href = canvas.toDataURL();
			link.setAttribute(
				'download',
				'youridol.png',
			);
			link.setAttribute('crossorigin', 'anonymous');
			link.click();
		}
	}

	return (
		<div className='Buttons1'>
			<div className='Buttonblocks'>
				<span> Background </span>
				<input type="checkbox" checked={background} onChange={handleBackground}></input>
				<span> Brush size : </span>
				<input type="range" id="paintRange" min="0.1" max="10" value={brushsize} step="0.1" onChange={handleBrush}/>
				<span> Eraser size : </span>
				<input type="range" id="eraseRange" min="0.1" max="100" value={erasersize} step="0.1" onChange={handleEraser}/>
			</div>
			<div className='Buttonblocks'>
				<button className={(realtime ? "bg-blue-500 hover:bg-blue-400 border-blue-700 hover:border-blue-500 text-white font-bold py-2 px-4 border-b-4  rounded" : "bg-red-500 hover:bg-red-400 border-red-700 hover:border-red-500 text-white font-bold py-2 px-4 border-b-4  rounded")} onClick={handleRealtime}>
				REALTIME { realtime ? <div> ON ✅</div>  : <div> OFF🚫</div>}
				</button>
				<button className="bg-blue-500 hover:bg-blue-400 text-white font-bold py-2 px-4 border-b-4 border-blue-700 hover:border-blue-500 rounded" onClick={convertCanvases}>
				 <div>CONVERT 💣</div>
				</button>
				<button className="bg-blue-500 hover:bg-blue-400 text-white font-bold py-2 px-4 border-b-4 border-blue-700 hover:border-blue-500 rounded" onClick={modeChange}>
					{drawingstate ?  <div>BRUSH 🪄</div> : <div>ERASER 🖱</div>}
				</button>
			</div>
			<div className='Buttonblocks'>
				<button className="bg-blue-500 hover:bg-blue-400 text-white font-bold py-2 px-4 border-b-4 border-blue-700 hover:border-blue-500 rounded" onClick={clearCanvas}>
					<div>CLEAR 🥡</div>
					</button>
				<button className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded inline-flex items-center" onClick={downloadhandler}>
					<div>Download</div>
					<svg className="fill-current w-4 h-4 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M13 8V2H7v6H2l8 8 8-8h-5zM0 18h20v2H0v-2z"/></svg>
				</button>
				<button className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded inline-flex items-center" >
					<div>GOOD LUCK 🍀</div>
				</button>
			</div>
		</div>
	);
}

export default Buttons;