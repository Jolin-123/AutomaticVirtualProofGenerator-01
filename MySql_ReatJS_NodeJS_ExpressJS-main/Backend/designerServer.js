// import multer from 'multer'
// import path from 'path'
const multer = require('multer')
const path = require('path')
const express = require('express')   // framwork
const mysql = require('mysql')
const cors = require('cors')
const app = express();

//2.createing ./create
app.use(express.json());
app.use(cors());

// create mysql db
const db = mysql.createConnection({
    host:"localhost",
    user: 'root',
    password: '',
    database:'crud' //database name
})


// ********************** 01/29/2024 for upload file-2 *********************
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        db(null, 'public/images')
    },
    filename: (req, file, cb) => {
        db(null, file.fieldname + "_" + Date.now() + path.extname(file.originalname));
    }
})

const upload = multer({
    storage: storage
})


//make sure connecting the DB
db.connect(err => {
    if (err) {
        console.error('Database connection error:', err);
    } else {
        console.log('Connected to the database');
    }
});


// ********************** 01/29/2024 for upload file-1 *********************
app.post('/upload', upload.single('image'), (req,res)=> {
    console.log(req.file)

})


app.listen(8088, () => {
    console.log("lisenting");
})
