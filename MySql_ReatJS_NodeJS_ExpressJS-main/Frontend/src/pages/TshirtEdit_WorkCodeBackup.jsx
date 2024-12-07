import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './FileUploadStyle.css';
import { useNavigate, useParams } from 'react-router-dom';
// the post the revised data

    function TshirtEdit() {
        const { idx } = useParams(); // Extract idx from URL params
        const [data, setData] = useState(null); 
        const [style, setStyle] = useState('');
        const [color, setColor] = useState('');
        const [description, setDesc] = useState('');
        const [price, setPrice] = useState('');
        const [file, setFile] = useState(null);
        //desc
    
        const handleFile = (e) => {
            setFile(e.target.files[0]);
        };
    
        useEffect(() => {
            fetchData();
        }, [idx]); // Fetch data whenever idx changes
    
        const fetchData = () => {
            axios.get(`http://localhost:3030/onetshirt/${idx}/`)
                .then(res => {
                    if (res.data) {
                        setData(res.data);
                        setStyle(res.data.style);
                        setColor(res.data.color);
                        setDesc(res.data.description);
                        setPrice(res.data.price);
                    }
                })
                .catch(err => console.log(err));
        };
    
        const handleUpdate = () => {        
            axios.put(`http://localhost:3030/updatetshirt/${idx}/`,  { idx, style, color, description, price, file  })
                .then(res => {
                    if (res.data !== "") {
                        console.log("Update successful");
                       // fetchData(); // Refresh data after successful update
                       navigate("/comment");
                    } else {
                        console.log("Update failed");
                    }
                })
                .catch(err => console.log("Error during update: ", err));
        };
    
        //const navigate = useNavigate();
    
    const navigate = useNavigate();
    

    return (
        <div className='tshirt_container'>
          Analytics page
          <h4> Update T-shirt Information </h4>
          <div className='input_container' style={{backgroundColor: "green", color: "green"}}> 
            {/* Input fields to set style, color, description, and price */}
          
            <label htmlFor="style">Style: </label>
            <input type='text' placeholder='Style' value={style} onChange={(e) => setStyle(e.target.value)} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="color">Color: </label>
            <input type='text' placeholder='Color' value={color} onChange={(e) => setColor(e.target.value)} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="description">Description: </label>
            <input type='text' placeholder='Description' value={description} onChange={(e) => setDesc(e.target.value)} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="price">Price: </label>
            <input type='text' placeholder='Price' value={price} onChange={(e) => setPrice(e.target.value)} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />
            {/* File input for image */}
            <label htmlFor="imageInput">Choose Image:</label>
            <input id="imageInput" type='file' onChange={handleFile} style={{ marginBottom: '10px', marginLeft: '200px'}}/> 
            {/* Button to upload */}
            <button onClick={handleUpdate}>Upload</button>  <br/>
            {/* Display uploaded data */}
          </div>

          <div className='display_container'> 
              {data && (
                  <div>
                    <br/>
                    {/* Display image if available */}
                    {data.image && <img src={`http://localhost:3030/images/${data.image}`} alt="Uploaded" style={{ width: "350px", height: "300px" }} />}
                      
                      <br/> <br/>
                      <p><h4> Style: {data.style} </h4></p>    

                      <p><h4> Color: {data.color} </h4></p>            

                      <p><h4> Description: {data.description} </h4></p>

                      <p><h4> Price: {data.price} </h4></p>

                      <button >Edit</button>
                      <button >Delete</button>
                  </div>  

                  
              )}

          </div> 

        </div>
    );
}

export default TshirtEdit;






