module tileoff_assumed_shape_matrix_kernels
contains

  subroutine matrix_add_assumed_shape(a, b, c)
    real :: a(:, :), b(:, :), c(:, :)
    integer :: i, j

    !$tileoff parallel tile(16, 16) pack(a:device, b:device, c:device)
    do j = 1, size(c, 2)
      do i = 1, size(c, 1)
        c(i, j) = a(i, j) + b(i, j)
      end do
    end do

    !$tileoff update host(c)
    !$tileoff release all
  end subroutine

end module

