<?php
// database connection code
// $con = mysqli_connect('localhost', 'database_user', 'database_password','database');

$con = mysqli_connect('localhost', 'root', '', 'face mask');

// get the post records
$uname = $_POST['email'];
$psw = $_POST['password'];

// insert in database
$sql= "SELECT * FROM student_details WHERE email = '$uname' AND password = '$psw' ";
$result = mysqli_query($con,$sql);
$check = mysqli_fetch_array($result);
if(isset($check)){

    header('Location: due.php');
}
else{
  header('Location: login.html');
}

?>
