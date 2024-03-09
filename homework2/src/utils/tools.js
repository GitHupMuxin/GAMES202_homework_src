function getRotationPrecomputeL(precompute_L, rotationMatrix){

	let rotationInv = mat4.create();
	mat4.invert(rotationInv, rotationMatrix);
	let MathRotation = mat4Matrix2mathMatrix(rotationInv);

	let SH1RotationMatrix = computeSquareMatrix_3by3(MathRotation);
	let SH2RotationMatrix = computeSquareMatrix_5by5(MathRotation);

	let result = [];
	for (let i = 0; i < 9; i++)
		result[i] = [];

	for (let i = 0; i < 3; i++)
	{
		let SH1Matrix = math.matrix(
									[precompute_L[1][i], precompute_L[2][i], precompute_L[3][i]],
		);
		let SH2Matrix = math.matrix(
									[precompute_L[4][i], precompute_L[5][i], precompute_L[6][i], precompute_L[7][i], precompute_L[8][i]],
		);
		let SH1 = math.multiply(SH1Matrix, SH1RotationMatrix);
		let SH2 = math.multiply(SH2Matrix, SH2RotationMatrix);
		result[0][i] = precompute_L[0][i];
		result[1][i] = SH1._data[0];
		result[2][i] = SH1._data[1];
		result[3][i] = SH1._data[2];

		result[4][i] = SH2._data[0];
		result[5][i] = SH2._data[1];
		result[6][i] = SH2._data[2];
		result[7][i] = SH2._data[3];
		result[8][i] = SH2._data[4];
	}
	return result;
}

function computeSquareMatrix_3by3(rotationMatrix){ // 计算方阵SA(-1) 3*3 
	
	// 1、pick ni - {ni}
	let n1 = [1, 0, 0, 0]; let n2 = [0, 0, 1, 0]; let n3 = [0, 1, 0, 0];

	// 2、{P(ni)} - A  A_inverse
	let p1 = SHEval(n1[0], n1[1], n1[2], 3);
	let p2 = SHEval(n2[0], n2[1], n2[2], 3);
	let p3 = SHEval(n3[0], n3[1], n3[2], 3);

	let A = math.matrix([
							[p1[1], p2[1], p3[1]],
							[p1[2], p2[2], p3[2]],
							[p1[3], p2[3], p3[3]],
	]);
	let A_inverse = math.inv(A);

	// 3、用 R 旋转 ni - {R(ni)}

	let n1_r = math.multiply(rotationMatrix, n1);
	let n2_r = math.multiply(rotationMatrix, n2); 
	let n3_r = math.multiply(rotationMatrix, n3);


	// 4、R(ni) SH投影 - S

	let pr1 = SHEval(n1_r[0], n1_r[1], n1_r[2], 3);
	let pr2 = SHEval(n2_r[0], n2_r[1], n2_r[2], 3);
	let pr3 = SHEval(n3_r[0], n3_r[1], n3_r[2], 3);

	let S = math.matrix([
						[pr1[1], pr2[1], pr3[1]],
						[pr1[2], pr2[2], pr3[2]],
						[pr1[3], pr2[3], pr3[3]]
	]);
	// 5、S*A_inverse

	let M = math.multiply(S, A_inverse);

	return M;
}

function computeSquareMatrix_5by5(rotationMatrix){ // 计算方阵SA(-1) 5*5
	
	// 1、pick ni - {ni}

	let k = 1 / math.sqrt(2);
	let n1 = [1, 0, 0, 0]; let n2 = [0, 0, 1, 0]; let n3 = [k, k, 0, 0]; 
	let n4 = [k, 0, k, 0]; let n5 = [0, k, k, 0];

	// 2、{P(ni)} - A  A_inverse
	
	let p1 = SHEval(n1[0], n1[1], n1[2], 3);
	let p2 = SHEval(n2[0], n2[1], n2[2], 3);
	let p3 = SHEval(n3[0], n3[1], n3[2], 3);
	let p4 = SHEval(n4[0], n4[1], n4[2], 3);
	let p5 = SHEval(n5[0], n5[1], n5[2], 3);

	let A = math.matrix([
						[p1[4], p2[4], p3[4], p4[4], p5[4]],
						[p1[5], p2[5], p3[5], p4[5], p5[5]],
						[p1[6], p2[6], p3[6], p4[6], p5[6]],
						[p1[7], p2[7], p3[7], p4[7], p5[7]],
						[p1[8], p2[8], p3[8], p4[8], p5[8]],
	])

	let A_inverse = math.inv(A);

	// 3、用 R 旋转 ni - {R(ni)}
	let n1_r = math.multiply(rotationMatrix, n1);
	let n2_r = math.multiply(rotationMatrix, n2);
	let n3_r = math.multiply(rotationMatrix, n3);
	let n4_r = math.multiply(rotationMatrix, n4);
	let n5_r = math.multiply(rotationMatrix, n5);

	// 4、R(ni) SH投影 - S
	let pr1 = SHEval(n1_r[0], n1_r[1], n1_r[2], 3);
	let pr2 = SHEval(n2_r[0], n2_r[1], n2_r[2], 3);
	let pr3 = SHEval(n3_r[0], n3_r[1], n3_r[2], 3);
	let pr4 = SHEval(n4_r[0], n4_r[1], n4_r[2], 3);
	let pr5 = SHEval(n5_r[0], n5_r[1], n5_r[2], 3);

	let S = math.matrix([
		[pr1[4], pr2[4], pr3[4], pr4[4], pr5[4]],
		[pr1[5], pr2[5], pr3[5], pr4[5], pr5[5]],
		[pr1[6], pr2[6], pr3[6], pr4[6], pr5[6]],
		[pr1[7], pr2[7], pr3[7], pr4[7], pr5[7]],
		[pr1[8], pr2[8], pr3[8], pr4[8], pr5[8]],
	])

	// 5、S*A_inverse
	let M = math.multiply(S, A_inverse);
	return M;
}

function mat4Matrix2mathMatrix(rotationMatrix){

	let mathMatrix = [];
	for(let i = 0; i < 4; i++){
		let r = [];
		for(let j = 0; j < 4; j++){
			r.push(rotationMatrix[i*4+j]);
		}
		mathMatrix.push(r);
	}
	//return math.matrix(mathMatrix)
	return math.transpose(mathMatrix)
}

function getMat3ValueFromRGB(precomputeL){

    let colorMat3 = [];
    for(var i = 0; i<3; i++){
        colorMat3[i] = mat3.fromValues( precomputeL[0][i], precomputeL[1][i], precomputeL[2][i],
										precomputeL[3][i], precomputeL[4][i], precomputeL[5][i],
										precomputeL[6][i], precomputeL[7][i], precomputeL[8][i] ); 
	}
    return colorMat3;
}