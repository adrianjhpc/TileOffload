module tileoff_matrix_add_2d_kernel
contains

  subroutine matrix_add_2d_prepare(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$tileoff update device(a)
    !$tileoff update device(b)
    !$tileoff update device(c)
  end subroutine

  subroutine matrix_add_2d_compute(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)
    integer :: i, j

    !$tileoff parallel tile(32, 32) pack(a:device, b:device, c:device)
    do j = 1, size(c, 2)
      do i = 1, size(c, 1)
        c(i, j) = a(i, j) + b(i, j)
      end do
    end do
  end subroutine

  subroutine matrix_add_2d_fetch(c)
    real :: c(:, :)

    !$tileoff update host(c)
  end subroutine

  subroutine matrix_add_2d_release(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)

    !$tileoff release(a, b, c)
  end subroutine

end module

