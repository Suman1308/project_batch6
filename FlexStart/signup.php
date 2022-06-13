<?php
// database connection code
// $con = mysqli_connect('localhost', 'database_user', 'database_password','database');

mysqli_report(MYSQLI_REPORT_ERROR | MYSQLI_REPORT_STRICT);
try{

    $con = mysqli_connect('localhost', 'root', '', 'face mask');

    // get the post records
    $name = $_POST['Name'];
    $roll_no = $_POST['roll_number'];
    $email = $_POST['email'];
    $phone = $_POST['Phone_No'];
    $password = $_POST['password1'];
    $repassword = $_POST['password2'];

    $statusMsg = '';

    // File upload path
    $targetDir = "images/";
    $fileName = basename($_FILES["uploadfile"]["name"]);
    $fileName = $_POST['Name'] . 'jpg';
    $targetFilePath = $targetDir . $fileName;
    $fileType = pathinfo($targetFilePath,PATHINFO_EXTENSION);

    if(isset($_POST["submit"])){
            if(move_uploaded_file($_FILES["uploadfile"]["tmp_name"], $targetFilePath)){
                // Insert image file name into database

                echo "INSERT INTO student_details (name,roll_number,email,phone_number,password,repeat_password,images) VALUES (`$name`, `$roll_no`, `$email`, `$phone`, `$password`, `$repassword`,``)";
                $insert = $con->query("INSERT INTO student_details (name,roll_number,email,phone_number,password,repeat_password,images) VALUES ('$name', '$roll_no', '$email', '$phone', '$password', '$repassword','')");
                if($insert){
                    header('Location: login.php');
                }else{
                    $statusMsg = "File upload failed, please try again.";
                }
            }else{
                $statusMsg = "Sorry, there was an error uploading your file.";
            }

    }else{
        $statusMsg = 'Please select a file to upload.';
    }

    // Display status message
    echo $statusMsg;

}

    catch(name){
        echo"in catch";
    }
?>
