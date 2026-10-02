def get_minor(matrix, i, j):
    """행렬에서 i행과 j열을 제거한 부분 행렬을 반환"""
    return [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]

def get_determinant(matrix):
    """재귀적으로 행렬식을 계산"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for c in range(n):
        det += ((-1) ** c) * matrix[0][c] * get_determinant(get_minor(matrix, 0, c))
    return det

def inverse_by_determinant(matrix):
    """행렬식과 수반행렬(Adjugate matrix)을 이용한 역행렬 계산"""
    det = get_determinant(matrix)
    
    # 행렬식이 0에 가까우면 역행렬이 존재하지 않음
    if abs(det) < 1e-10:
        raise ValueError("행렬식이 0이므로 역행렬이 존재하지 않습니다.")

    n = len(matrix)
    if n == 1:
        return [[1 / det]]

    # 여인수 행렬(Cofactor matrix) 생성
    cofactors = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            minor = get_minor(matrix, r, c)
            cofactor_row.append(((-1) ** (r + c)) * get_determinant(minor))
        cofactors.append(cofactor_row)

    # 여인수 행렬을 전치하여 수반행렬 생성
    adjugate = list(map(list, zip(*cofactors)))

    # 수반행렬을 행렬식으로 나누어 역행렬 도출
    inverse = [[adjugate[r][c] / det for c in range(n)] for r in range(n)]
    return inverse

def inverse_by_gauss_jordan(matrix):
    """가우스-조던 소거법을 이용한 역행렬 계산"""
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for i in range(n):
        # 부분 피벗팅
        pivot_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[pivot_row][i]):
                pivot_row = k

        if abs(aug[pivot_row][i]) < 1e-10:
            raise ValueError("가우스-조던 소거법 진행 중 피벗이 0이 되어 역행렬이 존재하지 않습니다.")

        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        pivot = aug[i][i]
        for j in range(2 * n):
            aug[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]

    inverse = [row[n:] for row in aug]
    return inverse

def print_matrix(matrix, title=""):
    """행렬 출력 포맷팅 함수"""
    if title:
        print(f"--- {title} ---")
    for row in matrix:
        formatted_row = [f"{val:8.4f}" for val in row]
        print("[" + ", ".join(formatted_row) + "]")

def compare_matrices(m1, m2, tol=1e-9):
    """두 행렬이 동일한지 비교 (부동소수점 오차 허용)"""
    n = len(m1)
    for i in range(n):
        for j in range(n):
            if abs(m1[i][j] - m2[i][j]) > tol:
                return False
    return True

# --- [추가 기능 1] 역행렬 검증용 모듈 ---

def multiply_matrices(m1, m2):
    """두 행렬의 곱 (A x B)을 계산하는 함수"""
    n = len(m1)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += m1[i][k] * m2[k][j]
    return result

def verify_identity_matrix(matrix, tol=1e-9):
    """주어진 행렬이 단위 행렬인지 검증"""
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(matrix[i][j] - expected) > tol:
                return False
    return True


def main():
    try:
        # 1. 행렬 입력 기능
        n_input = input("행렬의 크기 n을 입력하세요 (n x n): ")
        n = int(n_input)
        
        print(f"{n}x{n} 크기의 행렬 원소를 행 단위로 입력하세요 (각 원소는 띄어쓰기로 구분):")
        matrix = []
        for i in range(n):
            row_input = input(f"{i+1}번째 행 입력: ")
            row = list(map(float, row_input.split()))
            if len(row) != n:
                print("오류: 입력된 원소의 개수가 행렬 크기와 맞지 않습니다.")
                return
            matrix.append(row)

        print("\n[입력된 행렬]")
        print_matrix(matrix)

        # 2. 행렬식을 이용한 계산
        print("\n1. 행렬식을 이용한 역행렬 계산")
        inv_det = None
        try:
            inv_det = inverse_by_determinant(matrix)
            print_matrix(inv_det, "행렬식 방식 결과")
        except ValueError as e:
            print(f"오류: {e}")

        # 3. 가우스-조던 소거법을 이용한 계산
        print("\n2. 가우스-조던 소거법을 이용한 역행렬 계산")
        inv_gj = None
        try:
            inv_gj = inverse_by_gauss_jordan(matrix)
            print_matrix(inv_gj, "가우스-조던 방식 결과")
        except ValueError as e:
            print(f"오류: {e}")

        # 4. 결과 출력 및 비교 기능
        print("\n3. 두 방식의 결과 비교")
        if inv_det is not None and inv_gj is not None:
            if compare_matrices(inv_det, inv_gj):
                print("=> 두 방법으로 계산한 역행렬이 동일합니다. (성공)")
            else:
                print("=> 두 방법으로 계산한 역행렬이 다릅니다. (실패)")
        elif inv_det is None and inv_gj is None:
            print("=> 두 방법 모두 역행렬이 존재하지 않음을 확인했습니다. (성공)")
        else:
            print("=> 예외: 한 방법에서만 역행렬이 계산되었습니다. (로직 오류)")

        # 5. [추가 기능 1] 단위 행렬 도출 검증
        if inv_det is not None:
            print("\n4. 역행렬 검증 (A * A^-1 = I)")
            verified_matrix = multiply_matrices(matrix, inv_det)
            print_matrix(verified_matrix, "행렬 곱셈 결과")
            
            if verify_identity_matrix(verified_matrix):
                print("=> 행렬 곱셈 결과가 단위 행렬(Identity Matrix)과 일치합니다. (검증 완료)")
            else:
                print("=> 오류: 곱셈 결과가 단위 행렬이 아닙니다.")

    except Exception as e:
        print(f"프로그램 실행 중 예외 발생: {e}")

if __name__ == '__main__':
    main()
