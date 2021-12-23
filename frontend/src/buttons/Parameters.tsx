import '../styles/styles.css';

interface InputProps {
	canvasRef: React.RefObject<HTMLCanvasElement>;
	paramT: number;
	realtime: boolean;
	setParamT: React.Dispatch<React.SetStateAction<number>>;
	outputRef: React.RefObject<HTMLCanvasElement>;
	fetchImage: () => void;
}

function Parameters({canvasRef, paramT, realtime, setParamT, fetchImage} : InputProps) {
	// Brush bar handling
	const handleParamT = (event: React.ChangeEvent<HTMLInputElement>) : number | undefined => {
		if (!canvasRef.current) { return; }
		const size:string = event.target.value;
		setParamT(+size);
		if (realtime === true) { fetchImage(); };
	}

	return (
		<div className='Buttons2'>
			<div className='Parameters'>
				<span> Parameter T :
					<input type="range" id="paintRange" min="0.01" max="1.0" value={paramT} step="0.01" onChange={handleParamT}/>
				</span>
			</div>
		</div>
	);
}

export default Parameters;