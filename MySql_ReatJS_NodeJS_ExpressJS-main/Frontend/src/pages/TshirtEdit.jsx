
// the post the revised data
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './FileUploadStyle.css';
import { useNavigate, useParams } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
    
    function TshirtEdit() {
        
        const [data, setData] = useState(null); 
        const [style, setStyle] = useState('');
        const [color, setColor] = useState('');
        const [description, setDesc] = useState('');
        const [price, setPrice] = useState('');
        const [file, setFile] = useState(''); // State variable to store the selected image file
        const [idx, setIdx] = useState(0); // State variable to store the filename
        const { idx: RouterId } = useParams(); // Extract idx from URL params
    
    
        const handleFile = (e) => {
            setFile(e.target.files[0]);
            console.log("after setFile image name: : ", e.target.files[0])
        };
     
    
        useEffect(() => {
            fetchData();
            setIdx(RouterId);
            console.log("Tshirt id is: " , RouterId);
        }, [RouterId]); // Fetch data whenever idx changes
    
        const fetchData = () => {
            axios.get(`http://localhost:3030/onetshirt/${RouterId}/`)
                .then(res => {
                    if (res.data) {
                        setData(res.data);
                        setIdx(res.data.idx);
                        setStyle(res.data.style);
                        setColor(res.data.color);
                        setDesc(res.data.description);
                        setPrice(res.data.price);
                        setFile(res.data.image);
                    }
                })
                .catch(err => console.log(err));
        };
    

        const handleUpdate = () => {
            const formData = new FormData();
            formData.append('style', style);
            formData.append('color', color);
            formData.append('description', description);
            formData.append('price', price);
            formData.append('image', file);
        
            axios.put(`http://localhost:3030/updatetshirt/${idx}/`,  formData)
                .then(res => {
                    if (res.data !== "") {
                        console.log("Update successful front end");
                        fetchData(); // Refresh data after successful update
                       //navigate("/comment");
                    } else {
                        console.log("Update failed");
                    }
                })
                .catch(err => console.log("Error during update: ", err));
        };
    

    return (
        <Sidebar>
        {/* Analytics page */}
        <div className='tshirt_container'>

          <div className='input_container' style={{ color: "Black"}}> 
            {/* Input fields to set style, color, description, and price */}
            <div>
                 <h2 style={{color: "green", margin: '10px 0 30px'}}> Update Tshirt </h2>
            </div>
        
            
            <label htmlFor="style"> Style: </label>
            <input type='text' placeholder='Style' value={style} onChange={(e) => setStyle(e.target.value.toUpperCase())} style={{ marginBottom: '10px', marginLeft:'5px' }} />
            <br />

            <label htmlFor="color">Color: </label>
            <input type='text' placeholder='Color' value={color} onChange={(e) => setColor(e.target.value.toUpperCase())} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="description">Description: </label>
            <input type='text' placeholder='Description' value={description} onChange={(e) => setDesc(e.target.value.toUpperCase())} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />

            <label htmlFor="price">Price: </label>
            <input type='text' placeholder='Price' value={price} onChange={(e) => setPrice(e.target.value.toUpperCase())} style={{ marginBottom: '10px',  marginLeft:'5px' }} />
            <br />
            {/* File input for image */}
            <div> 
            <label htmlFor="imageInput">Choose Image:</label>
            <input id="imageInput" type='file' onChange={handleFile} style={{ margin: '15px 0 40px 0', marginLeft: '200px'}}/> 
            </div>
           
            {/* Button to upload */}
            <button onClick={handleUpdate} className='btn-update'> Upload </button>  <br/>
            {/* Display uploaded data */}
          </div>

          <div className='display_container'> 
              {data && (
                  <div>
                    <br/>
                    {/* Display image if available */}
                    {data.image && <img src={`http://localhost:3030/images/${data.image}`} alt="Uploaded" style={{ width: "300px", height: "350px" }} />}
                      
                      <br/> <br/>
                      <p><h4> Style: {data.style} </h4></p>    

                      <p><h4> Color: {data.color} </h4></p>            

                      <p><h4> Description: {data.description} </h4></p>

                      <p><h4> Price: {data.price} </h4></p>

                      {/* <button >Edit</button>
                      <button >Delete</button> */}
                  </div>  
              )}

          </div> 

        </div>
        </Sidebar>
    );
}

export default TshirtEdit;





