module tileoff_matmul_2d_tf32_kernel
contains

  subroutine matmul_2d_tf32_prepare(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$tileoff enter data copyin(a, b) create(c)
  end subroutine

  subroutine matmul_2d_tf32_compute(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)
    integer :: i, j, p
    real :: acc

    !$tileoff parallel tile(64, 64, 32) pack(a:device, b:device, c:device) matmul_precision(tf32)
    do j = 1, size(c, 2)
      do i = 1, size(c, 1)
        acc = 0.0
        do p = 1, size(a, 2)
          acc = acc + a(i, p) * b(p, j)
        end do
        c(i, j) = acc
      end do
    end do

  end subroutine

  subroutine matmul_2d_tf32_fetch(c)
    real :: c(:, :)

    !$tileoff update host(c)
  end subroutine

  subroutine matmul_2d_tf32_release(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$tileoff exit data delete(a, b, c)
  end subroutine

end module

