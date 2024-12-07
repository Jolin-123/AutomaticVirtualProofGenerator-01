//new SERVER code
// ********************** 01/29/2024 for upload file-3 *********************
// ********************** 02/9/2024 for upload for backend CURD *********************
import multer from 'multer'
import path from 'path'
import express from 'express'
import mysql from 'mysql'
import cors from 'cors'


const app = express();

//2.createing ./create
app.use(express.json());
app.use(cors());

//for showing image
app.use(express.static('public'));


// create mysql db
const db = mysql.createConnection({
    host:"localhost",
    user: 'root',
    password: '',
    database:'productlist' //database name

})




// ********************** 01/29/2024 for upload file-2 *********************


//make sure connecting the DB
db.connect(err => {
    if (err) {
        console.error('Database connection error:', err);
    } else {
        console.log('Connected to the database');
    }
});



// ********************** 02/12/2024 for Sign up *********************

app.post('/signup', (req, res) => {
    //const { name, email, password } = req.body;
    const sql = "INSERT INTO usersignup (`name`, `email`, `password`) VALUES (?, ?, ?)";
    //const values = [name, email, password]; // Array of values
    const values = [req.body.name, req.body.email, req.body.password]; 

    db.query(sql, values, (err, data) => {
        if (err) {
            console.error("Error posting signup info :", err);
            return res.status(500).json({ message: "Error posting signup info." });
        }
        return res.json(data);
    });
});

// Login 
app.post('/login', (req, res) => {
    const sql = "SELECT * FROM usersignup";
    const values = [
        req.body.email,
        req.body.password
    ];
    db.query(sql, values, (err, data) => {
        if (err) {
            console.error("Error occurred while querying database:", err);
            return res.status(500).json({ message: "An error occurred while logging in. Please try again later." });
        }
        let loginSuccess = false;
        data.forEach(user => {
            if (user.email === req.body.email && user.password === req.body.password) {
                loginSuccess = true;
                console.log("after compare each data in database backedn.")
            }
        });
        if (loginSuccess) {
            return res.json("Success"); // Login successful
        } else {
            return res.json("Fail");
        }
    });
});



// ==================================================================================================
// ********************** 01/29/2024 for upload file-1 *********************
// ================================== Productlistv Table CURD Method ================================

app.get('/', (re, res)=> {
    return res.json("Back end work for product");
})

//get all data from productlistv 
app.get('/productlistv', (req, res) => {
    const sql = "SELECT * FROM productlistv";
    db.query(sql, (err, data) => {
        if(err) return res.json("Error product @@ ");
        return res.json(data);
    });
});

// get total orders in productlistv
app.get('/productlistv/count', (req, res) => {
    const sql = "SELECT COUNT(*) AS total FROM productlistv";
    db.query(sql, (err, data) => {
        if (err) {
            return res.json({ error: "Error counting products" });
        }
        return res.json({ total: data[0].total });
    });
});


// get all 3 tables data 
//Get one row from productlistv   good and work 
app.get('/allone/:ordernumber', (req, res) => {
    const ordernumber = req.params.ordernumber;
    const sql = `
        SELECT *
        FROM productlistv
        JOIN frontlogos ON productlistv.gid = frontlogos.id
        JOIN tshirtinfo ON productlistv.bid = tshirtinfo.idx
        WHERE productlistv.ord = ?
    `;
    db.query(sql, [ordernumber], (err, data) => {
        if(err) {
            console.error('Error executing query:', err);
            return res.status(500).json({ error: 'Internal Server Error' });
        }
        return res.json(data[0]); // Assuming you only want to return the first matching row
    });
});



//Get one row from productlistv
app.get('/oneOrder/:oid', (req, res) => {
    const orderId = req.params.oid;
    const sql = "SELECT * FROM productlistv WHERE oid = ?";
    db.query(sql, [orderId], (err, data) => {
        if (err) {
            console.error("Error order retrieve database. ", err);
            return res.status(500).json({ Message: "Error retrieving an order backend" });
        }
        if (data.length === 0) {
            return res.status(404).json({ Message: "No order  found for the given ID backend " });
        }

        const currentRow = data[0];
        console.log("Returned order object  from backend:", currentRow);
        return res.json(currentRow);
    });
});



//2. Post an order data to Productlistv
app.post('/create', (req, res) => {
    const sql = "INSERT INTO productlistv (ord, customer, label, bid, gid, bcolor, team, bdesc, xs, s, m, l, xl, xxl, xxxl, ttl, start, end) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    const values = [
        req.body.ord,
        req.body.customer,
        req.body.label,
        req.body.bid,
        req.body.gid,
        req.body.bcolor,
        req.body.team,
        req.body.bdesc,
        req.body.xs,
        req.body.s,
        req.body.m,
        req.body.l,
        req.body.xl,
        req.body.xxl,
        req.body.xxxl,
        req.body.ttl,
        req.body.start,
        req.body.end
    ];

    db.query(sql, values, (err, data) => {
        if (err) return res.json("Error inserting data into the database .9.");
        return res.json(data);
    });
});


//3.PUT Update an order information in productlistv
app.put('/update/:oid', (req, res) => {
    const sql = "UPDATE productlistv SET `ord`=?, `customer`=?, `label`=?, `bid`=?, `gid`=?, `bcolor`=?, `team`=?, `bdesc`=?, `xs`=?, `s`=?, `m`=?, `l`=?, `xl`=?, `xxl`=?, `xxxl`=?, `ttl`=?, `start`=?, `end`=? WHERE oid = ?";
    
    const values = [
        req.body.ord,
        req.body.customer,
        req.body.label,
        req.body.bid,
        req.body.gid,
        req.body.bcolor,
        req.body.team,
        req.body.bdesc,
        req.body.xs,
        req.body.s,
        req.body.m,
        req.body.l,
        req.body.xl,
        req.body.xxl,
        req.body.xxxl,
        req.body.ttl,
        req.body.start,
        req.body.end
    ];
    const productId = req.params.oid;

    db.query(sql, [...values, productId], (err, data) => {
        if (err) return res.json("Error putting data");
        return res.json(data);
    });
});


// Delete one order from productlistv by order id that from user click  //api,url
app.delete('/deleteProductList/:oid', (req, res) => {  
    const itemId = req.params.oid;
    const sql = "DELETE FROM productlistv WHERE oid = ?";
    
    //modele.js
    db.query(sql, [itemId], (err, result) => {
        if (err) {
            console.error("Error from delete prodocutList.", err);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            console.log("Deleted user with ID:", itemId);
            res.json({ message: 'User deleted successfully' });
        }
    });
});




// ================================== T-shirt Table CURD Method =====================================
// *********************************************************************************
// ********************** 02/26/2024 for upload t-shirt-1 image ********************
// Adding a new T-shirt to the tshirtinfo, including the image
const storage = multer.diskStorage({
    destination: (req, file, db) => {
        db(null, 'public/images')
    },
    filename: (req, file, db) => {
        db(null, file.fieldname + "_" + Date.now() + path.extname(file.originalname));
    }
})
const upload = multer({
    storage: storage
})

// ********************** 02/26/2024 for upload t-shirt-2 image *********************
// Add a new order into tshirtinfo
app.post('/upload', upload.single('image'), (req,res)=> {
    console.log(req.file);
    const image = req.file.filename;
    const { style, color, desc, price } = req.body; // Extract additional data
    const sql = "INSERT INTO tshirtinfo (image, style, color, description, price) VALUES (?, ?, ?, ?, ?)";
    db.query(sql, [image, style, color, desc, price], (err, result) => {
        if (err) {
            console.error("Error updating image:", err);
            return res.json({ Message: "Error" });
        }
        return res.json({ Status: "Success"});
    });
});


// Get the last order from tshirtinfo
app.get('/lastpost', (req, res) => {
    const sql = 'SELECT * FROM tshirtinfo ORDER BY idx DESC LIMIT 1'; // Query to select the last post
    db.query(sql, (err, result) => {
        if (err) {
            console.error("Error retrieving last post:", err);
            return res.status(500).json({ Message: "Error in geting code back from data@@" });
        }
        if (result.length === 0) {
            return res.status(404).json({ Message: "No posts found" });
        }
        // Extract the last post from the result array
        const lastPost = result[0];
        console.log("return object data from backend: " , lastPost);
        return res.json(lastPost);
        
    });
});


// Get all orders from tshirtinfo
app.get('/', (re, res)=> {
    const sql = 'select * from tshirt';
    db.query(sql, (err, result) => {
        if(err) return res.json("Error");
        return res.json(result);
    });
});




// ==================================================================================================
// ======================================= 3/10/2024 ================================================
// ================================== T-shirt Table CURD Method =====================================
//Get one order from tshirtinfo
app.get('/onetshirt/:idx', (req, res) => {
    const tshirtId = req.params.idx;
    const sql = "SELECT * FROM tshirtinfo WHERE idx = ?";
    db.query(sql, [tshirtId], (err, data) => {
        if (err) {
            console.error("Error retrieving logo table data:", err);
            return res.status(500).json({ Message: "Error retrieving Tshirt data from the backend" });
        }
        if (data.length === 0) {
            return res.status(404).json({ Message: "No tshirt posts found for the given ID backend " });
        }
        // Extract the data of the specified row
        // Pay attention to the data, one row only 
        const currentRow = data[0];
        console.log("Returned tshirt object data from backend:", currentRow);
        return res.json(currentRow);
    });
});




// Route to fetch all t-shirts
app.get('/alltshirt', (req, res) => {
    const sql = "SELECT * FROM tshirtInfo";
    db.query(sql, (err, data) => {
        if (err) {
            return res.status(500).json({ error: "Error retrieving data from tshirtInfo table" });
        }
        return res.json(data);
    });
});


// Route to delete one  T-shirt information from tshirtInfo
app.delete('/deleteTshirtInfo/:idx', (req, res) => {
    const itemId = req.params.idx;
    const sql = "DELETE FROM tshirtInfo WHERE idx = ?";

    //modele.js
    db.query(sql, [itemId], (err, result) => {
        if (err) {
            console.error("Error from delete tshirtInfo.", err);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            console.log("Deleted tshirtInfo  ID:", itemId);
            res.json({ message: 'tshirtInfo deleted successfully' });
        }
    });
});




// Update T-shirt information from tshirtInfo, including the image
const storageNew = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'public/images');
    },
    filename: (req, file, cb) => {
        cb(null, file.fieldname + "_" + Date.now() + path.extname(file.originalname));
    }
});

const editImage = multer({
    storage: storageNew
});



app.put('/updatetshirt/:idx', editImage.single('image'), (req, res) => {

    console.log("BACKEND UpdateShirt code here$$$$, and id",req.file);
    console.log("BACKEND UpdateShirt code  id",req.idx);
    if (!req.file) {
        return res.status(400).json({ message: "No tshirt image uploaded backend**" });
    }

    const sql = "UPDATE tshirtInfo SET `style`=?, `color`=?, `description`=?, `price`=?, `image`=? WHERE idx = ?";
    //console.log(req.file);
   
    const { style, color, description, price } = req.body;
    const image = req.file.filename;
    const tshirtItemId = req.params.idx;

    const values = [
        
        style,
        color,
        description,
        price,
        image,
        tshirtItemId
        
    ];

    db.query(sql, values, (err, data) => {
        if (err) {
            console.error("Error updating tshirt data:", err);
            return res.json({ Message: "Error in update tshirt info backend" });
        }
        return res.json({ Status: "Success" });
    });
});


//********************************************************************************************
//******************************** 3/1/2024 Logos Upload Methods *****************************
// Frontlogos Table
// work fine for posting a new logo
// Post one order into frontlogos (inlucing the image)
const storageLogo = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'public/images');
    },
    filename: (req, file, cb) => {
        cb(null, file.fieldname + "_" + Date.now() + path.extname(file.originalname));
    }
});

const logoImage = multer({
    storage: storageLogo
});

app.post('/uploadlogo', logoImage.single('imagel'), (req, res) => {
    console.log("here is req file: ", req.file);
    const imagel = req.file.filename; // Changed variable name from image to imagel
    const { gname, logodesc, tname, width, height, c1, c2, c3, c4, c5, c6 } = req.body;
    const sql = "INSERT INTO frontlogos (imagel, gname, logodesc, tname, width, height, c1, c2, c3, c4, c5, c6 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

    db.query(sql, [imagel, gname, logodesc, tname, width, height, c1, c2, c3, c4, c5, c6], (err, result) => {
        if (err) {
            console.error("Error updating logo image:", err);
            return res.json({ Message: "Error" });
        }
        return res.json({ Status: "Success"});
    });
});



// Get the last logo from frontlogos
app.get('/logolastpost', (req, res) => {
    const sql = 'SELECT * FROM frontlogos ORDER BY id DESC LIMIT 1'; // Query to select the last post
    db.query(sql, (err, result) => {
        if (err) {
            console.error("Error retrieving logo table last post:", err);
            return res.status(500).json({ Message: "Error in geting logo info from back  end **" });
        }
        if (result.length === 0) {
            return res.status(404).json({ Message: "No logo posts found" });
        }
        // Extract the last post from the result array
        const logolastpost = result[0];
        console.log("return logo object data from backend: " , logolastpost);
        return res.json(logolastpost);
        
    });
});


//Get one logo from frontlogos
app.get('/logoCurrent/:id', (req, res) => {
    const itemId = req.params.id;
    const sql = 'SELECT * FROM frontlogos WHERE id = ?'; // Query to select the specific row by ID

    db.query(sql, [itemId], (err, result) => {
        if (err) {
            console.error("Error retrieving logo table data:", err);
            return res.status(500).json({ Message: "Error retrieving logo table data from the backend" });
        }
        if (result.length === 0) {
            return res.status(404).json({ Message: "No logo posts found for the given ID backend " });
        }
        // Extract the data of the specified row
        const currentRow = result[0];
        console.log("Returned logo object data from backend:", currentRow);
        return res.json(currentRow);
    });
});


//Get all logos from frontlogos
// work for know but image is not display 
app.get('/alllogo', (req, res) => {
    const sql = "SELECT * FROM frontlogos";
    db.query(sql, (err, data) => {
        if(err) return res.json("Error in get all logo backend @@ ");
        return res.json(data);
    });
});


// Route to delete one LOGO information from frontlogos
app.delete('/deleteLogoList/:id', (req, res) => {
    const itemId = req.params.id;
    const sql = "DELETE FROM frontlogos WHERE id = ?";

    //modele.js
    db.query(sql, [itemId], (err, result) => {
        if (err) {
            console.error("Error from delete frontlogos.", err);
            res.status(500).json({ error: 'Internal server error' });
        } else {
            console.log("Deleted frontlogos  ID:", itemId);
            res.json({ message: 'frontlogos deleted successfully' });
        }
    });
});



// UPDATAE ONE LOGO 3/9/2024 continuous 
//hoping this part work fine for posting a new logo
//Update one logo information including the image
const storageLogoUpdate = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'public/images');
    },
    filename: (req, file, cb) => {
        cb(null, file.fieldname + "_" + Date.now() + path.extname(file.originalname));
    }

    //     filename: (req, file, cb) => {
//         cb(null, file.fieldname + "_" + Date.now() + path.extname(file.originalname));
//     }
});

const logoImageUpdate = multer({
    storage: storageLogoUpdate
});


app.put('/updatelogo/:id', logoImageUpdate.single('imagel'), (req, res) => {
    console.log(req.file);
    if (!req.file) {
        return res.status(400).json({ message: "No file uploaded backend**" });
    }
    const { gname, logodesc, tname, width, height, c1, c2, c3, c4, c5, c6 } = req.body;
    const imagel = req.file.filename;
    const itemId = req.params.id;
    // Implement logic to update logo data in the database
    db.query('UPDATE frontlogos SET gname=?, logodesc=?, tname=?, width=?, height=?, imagel=?, c1=?, c2=?, c3=?, c4=?, c5=?, c6=? WHERE id=?', 
        [gname, logodesc, tname, width, height, imagel,  c1, c2, c3, c4, c5, c6, itemId], 
        (err, result) => {
            if (err) {
                console.error("Error updating logo backend:", err);
                return res.status(500).json({ Status: "Error", Message: "Error updating logo backend." });
            }
            return res.json({ Status: "Success" });
        }
    );
});



app.listen(3030, () => {
    console.log("lisenting product list");
})


